# Basic module
import os
import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm
from model.parse_args_test import  parse_args
import re
from PIL import Image
import pandas as pd
from astropy.io import fits

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param
from model.dataloader import TestSetLoaderFits

# Model
from model.model_factory import get_model


def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt=load_dataset(args.root, args.dataset,args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor()])
        testset         = TestSetLoaderFits (dataset_dir, img_id=val_img_ids, preprocess=args.preprocess,
                                             base_size=args.base_size, crop_size=args.crop_size, transform=input_transform, suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset, batch_size=args.test_batch_size, num_workers=args.workers, drop_last=False)

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
        else:
            raise Exception('wrong model name')
        model = model.cuda()
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model = model

        # Initialize evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Load trained model
        checkpoint        = torch.load(args.model_dir)
        self.model.load_state_dict(checkpoint['state_dict'])

        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        # static snr
        bins = 15
        tp_count_array = np.zeros(bins + 1)
        fn_count_array = np.zeros(bins + 1)
        with torch.no_grad():
            fuse = np.zeros((4096, 4096))
            for _, (data, data_name) in enumerate(tbar):
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

            # # visualize
            # pred_fpath = os.path.join('dataset', args.dataset, 'visual_result')
            # recreate_dir(pred_fpath)
            # pred_fpath = os.path.join(pred_fpath, 'pred_fuse')
            # recreate_dir(pred_fpath)
            # pimg = Image.fromarray(fuse)
            # pimg.convert('L').save(os.path.join(pred_fpath, '20200821152001999_7824_824122_LVT04.png'))

            # get connected region proposals
            test_img = fits.getdata('dataset/fitstrans/fitsdata/4/20200821152001999_7824_824122_LVT04.fits')
            pred_region = measure.regionprops(measure.label(fuse, connectivity=2), intensity_image=test_img)
            # test ipd
            ipd_file = open('outfiles/testIPD/test.IPD', 'w')
            for pred in tqdm(pred_region):
                dataline = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
                    ("%.3f" % (pred.centroid_weighted[1])).zfill(8), ("%.3f" % (pred.centroid_weighted[0])).zfill(8),
                    "%04d" % pred.area,
                    "%04d" % (pred.bbox[3] - pred.bbox[1]), "%04d" % (pred.bbox[2] - pred.bbox[0]),
                    "0000000000",
                    "00000.000", "00000.000"
                )
                ipd_file.write(dataline)

            ipd_file.close()

            df_pred = pd.read_table('outfiles/testIPD/test.IPD', sep='\s+', header=None, encoding='utf-8',
                                    names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

            # eval with ground-truth file
            fname = os.path.join('dataset', self.args.dataset, 'ipdSNR', '20200821152001999_7824_824122_LVT04.IPD')
            df_gt = pd.read_table(fname, sep='\s+', header=None, encoding='utf-8',
                                  names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar', 'snr'])

            ''' evaluate by KD-Tree '''
            thres = 1.5
            all_pred = len(df_pred)
            all_gt = len(df_gt)

            kd_gt = KDTree(df_gt[['col', 'row']].values)
            match = kd_gt.query(df_pred[['col', 'row']].values, k=1)
            tp_idx_fromgt = np.unique(match[1][match[0] < thres])
            tp_objs = df_gt.copy().iloc[tp_idx_fromgt]
            fn_objs = df_gt.copy().drop(labels=tp_idx_fromgt)
            # static snr
            for _, tp_item in tp_objs.iterrows():
                snr_int = int(tp_item['snr'])
                if snr_int < 0:
                    print('a minus snr!')
                elif snr_int <= bins:
                    tp_count_array[snr_int] += 1
                elif snr_int > bins:
                    tp_count_array[bins] += 1
            for _, fn_item in fn_objs.iterrows():
                snr_int = int(fn_item['snr'])
                if snr_int < 0:
                    print('a minus snr!')
                elif snr_int <= bins:
                    fn_count_array[snr_int] += 1
                elif snr_int > bins:
                    fn_count_array[bins] += 1

            tp = len(np.unique(match[1][match[0] < thres]))
            fp = all_pred - tp
            fn = all_gt - tp
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision != 0 and recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
                faRate = (1 / precision - 1) * recall
            else:
                f1 = 0
                faRate = 0
                
            # print('tp-snr: {}'.format(tp_count_array))
            # print('fn-snr: {}'.format(fn_count_array))
            print('precision:{}, recall:{}, f1:{}, faRate:{}'.format(precision, recall, f1, faRate))


def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # set cuda device
    main(args)





