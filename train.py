# torch and visulization
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import load_dataset, load_param
from model.dataloader import TrainSetLoaderFits, TestSetLoaderFits

# model
from model.model_factory import get_model

import re
import pandas as pd
from astropy.io import fits
from scipy.spatial import KDTree

# random seed
seed_torch()

# ================= save model according recall =====================


class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, self.val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([transforms.ToTensor()])
        trainset = TrainSetLoaderFits(dataset_dir, img_id=train_img_ids, preprocess=args.preprocess, mask_type=args.mask_type,
                                      base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, 
                                      suffix=args.suffix, label_mode=args.label_mode)
        testset = TestSetLoaderFits(dataset_dir, img_id=self.val_img_ids, preprocess=args.preprocess, 
                                    base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, 
                                    suffix=args.suffix, label_mode=args.label_mode)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True,
                                     num_workers=args.workers, drop_last=True)
        self.test_data = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers,
                                    drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model.lower() == 'cdnnet':
            Res_CBAM_block, CDNNet = get_model(args.model)
            model = CDNNet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, 
                           nb_filter=nb_filter, netdepth=args.netdepth, scale_method=args.scale_method, 
                           deep_supervision=args.deep_supervision)
        elif args.model.lower() == 'dnanet':
            Res_CBAM_block, DNANet = get_model(args.model)
            model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
                                nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model.lower() == 'unet':
            Res_CBAM_block, UNet = get_model(args.model)
            model = UNet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
                                nb_filter=nb_filter, deep_supervision=False)
        else:
            raise Exception('wrong model name')
        model = model.cuda()
        ## Optimizing
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        ## Scheduling
        if args.scheduler == 'CosineAnnealingLR':
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        ## Initialing weights
        if args.resume_from.strip() == '':
            model.apply(weights_init_xavier)
            self.start_epoch = args.start_epoch
        else:
            model.load_model(args.resume_from)
            try:
                checkpoint = torch.load(args.resume_from, map_location=lambda storage, loc: storage)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.start_epoch = checkpoint['epoch']
                del checkpoint
            except Exception as e:
                print(e)
                print('model file does not contains optimizer or scheduler state, cannot resume training')
        print("Model Initializing")
        self.model = model

        # Evaluation metrics
        self.best_f1 = 0
        self.best_recall = 0
        self.test_fitsimg = fits.getdata("./dataset/fitstrans/fitsdata/4/20200821152001999_7824_824122_LVT04.fits")

    # Training
    def training(self, epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()

        for i, (data, labels) in enumerate(tbar):
            data = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds = self.model(data)
                loss = 0
                for pred in preds:
                    ''' 1. SIoU-loss '''
                    loss += siouLoss(pred, labels)
                    ''' 2. SFL-loss '''
                    loss += sflLoss(pred, labels) / 100    # for real: /100; for simu: /800
                loss /= len(preds)
            else:
                pred = self.model(data)
                loss = 0
                ''' 1. SIoU-loss '''
                loss += siouLoss(pred, labels)
                ''' 2. SFL-loss '''
                loss += sflLoss(pred, labels) / 100    # for real: /100; for simu: /800

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))

        self.train_loss = losses.avg
        save_train_loss = 'result/' + self.save_dir + '/' + self.save_prefix + '_training_loss.log'
        with open(save_train_loss, 'a') as f:
            f.write('Epoch:{}, loss:{}\n'.format(epoch, losses.avg))
        save_lr = 'result/' + self.save_dir + '/' + self.save_prefix + '_lr.log'
        with open(save_lr, 'a') as f:
            f.write('Epoch:{}, lr:{}\n'.format(epoch, self.scheduler.get_last_lr()))

    # Testing
    def testing(self, epoch):
        tbar = tqdm(self.test_data)

        self.model.eval()
        with torch.no_grad():
            # get predict detection
            fuse = np.zeros((4096, 4096))
            for i, (data, data_name) in enumerate(tbar):
                data = data.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    pred = preds[-1]
                else:
                    pred = self.model(data)
                pattern = re.compile('LVT04-(\d+)-(\d+)')
                for num in range(len(pred)):
                    up, left = map(int, pattern.search(data_name[num]).groups())
                    predsss = np.array((torch.sigmoid(pred[num]) > 0.4).cpu()).astype('int64') * 255
                    predsss = np.uint8(predsss)
                    fuse[left:left + 256, up:up + 256] = predsss[0]

            # get connected region proposals
            pred_region = measure.regionprops_table(measure.label(fuse, connectivity=2),
                                                    properties=('label', 'centroid_weighted'), intensity_image=self.test_fitsimg)
            pred_region_array = np.concatenate(
                [np.expand_dims(pred_region['centroid_weighted-0'], axis=1), np.expand_dims(pred_region['centroid_weighted-1'], axis=1),
                 np.expand_dims(pred_region['label'], axis=1)], axis=1)
            df_pred = pd.DataFrame(pred_region_array, columns=["row", "col", "label"])

            # eval with ground-truth file
            fname = "./dataset/fitstrans/fitsdata/4/20200821152001999_7824_824122_LVT04.IPD"
            labelfile = open(fname)
            for i in range(13):
                labelfile.readline()
            satnum = int(labelfile.readline().split()[-1])
            labelfile.close()
            df_gt = pd.read_table(fname, sep='\s+', header=None, skiprows=14 + satnum, encoding='utf-8',
                                  names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])
            # calculate tp, fp, fn
            thres = 1.5
            tp = 0
            all_pred = len(df_pred)
            all_gt = len(df_gt)

            ''' evaluate by KD-Tree '''
            kd_pred = KDTree(df_pred[['col', 'row']].values)
            match = kd_pred.query(df_gt[['col', 'row']].values, k=1)
            tp = len(np.unique(match[1][match[0] < thres]))
            fp = all_pred - tp
            fn = all_gt - tp
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision != 0 and recall != 0:
                f1 = 2*precision*recall/(precision+recall)
                faRate = (1 / precision - 1) * recall
            else:
                f1 = 0
                faRate = 0
            print('Epoch:{}, precision:{}, recall:{}, f1:{}, faRate:{}'.format(epoch, precision, recall, f1, faRate))
            
            # save result
            save_good_dir = 'result/' + self.save_dir + '/' + self.save_prefix + '_good_precision_recall.log'
            with open(save_good_dir, 'a') as f:
                f.write('Epoch:{}, precision:{}, recall:{}, f1:{}, faRate:{}\n'.format(epoch, precision, recall, f1, faRate))
            if precision > 0.8 and recall > self.best_recall:
                save_ckpt({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'recall': recall,
                    'precision': precision
                }, save_path='result/' + self.save_dir,
                    filename='recall_{}_precision_{}_'.format(recall, precision) + self.save_prefix + '_epoch{}'.format(
                        epoch) + '.pth.tar')
                self.best_recall = recall
            if precision > 0.45 and recall > 0.45 and f1 > self.best_f1:
                save_ckpt({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'recall': recall,
                    'precision': precision
                }, save_path='result/' + self.save_dir,
                    filename='recall_{}_precision_{}_'.format(recall, precision) + self.save_prefix + '_epoch{}'.format(
                        epoch) + '.pth.tar')
                self.best_f1 = f1


def main(args):
    trainer = Trainer(args)
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)
        trainer.scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # set cuda device
    main(args)





