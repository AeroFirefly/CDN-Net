from PIL import Image, ImageOps, ImageFilter
import platform, os
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from torch.nn import init
from datetime import datetime
import argparse
import shutil
from  matplotlib import pyplot as plt


class TrainSetLoader(Dataset):
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id ,base_size=512,crop_size=480,transform=None,suffix='.png',label_mode='seg'):
        super(TrainSetLoader, self).__init__()

        self.transform = transform
        self._items = img_id
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        self.label_mode = label_mode

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)
        return img, mask

    def __getitem__(self, idx):

        img_id     = self._items[idx]                        
        img_path   = self.images+'/'+img_id+self.suffix   
        if self.label_mode == "seg":
            label_path = self.masks +'/'+img_id+self.suffix
        elif self.label_mode == "centroid_mask" or self.label_mode == "centroid":
            label_path = img_id

        try:
            img = Image.open(img_path).convert('RGB')         
            if self.label_mode == "seg":
                mask = Image.open(label_path)
            elif self.label_mode == "centroid_mask" or self.label_mode == "centroid":
                file_idx = label_path.split("_")[0]
                frame_idx = label_path.split("_")[1]

                x = []
                y = []
                with open(os.path.join(self.masks, "data"+file_idx+".txt"), "r") as label_file:
                    for i, line in enumerate(label_file):
                        if i == int(frame_idx) + 1:
                            line_split = line.split()
                            for j in range(2, len(line_split)):
                                if (j-2) % 3 == 1:
                                    x.append(int(line_split[j]))
                                if (j-2) % 3 == 2:
                                    y.append(int(line_split[j]))
                            break

                mask = np.zeros(img.size).astype('uint8').transpose()
                for i in range(len(x)):
                    if self.label_mode == "centroid_mask":
                        x_range = [max(x[i]-1,0), min(x[i]+1,img.width)+1]
                        y_range = [max(y[i]-1,0), min(y[i]+1,img.height)+1]
                        mask[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 255
                    elif self.label_mode == "centroid":
                        mask[y[i], x[i]] = 255
                mask = Image.fromarray(mask)


            # # =============== test point label ================
            # mask_tmp = np.zeros(mask.size).astype('uint8').transpose()
            #
            # from skimage import measure
            # label = measure.label(np.array(mask), connectivity=2)
            # coord_label = measure.regionprops(label)
            # for i in range(len(coord_label)):
            #     centroid_label = np.array(list(coord_label[i].centroid))
            #     centroid_x = np.rint(centroid_label[0]).astype('int64')
            #     centroid_y = np.rint(centroid_label[1]).astype('int64')
            #     mask_tmp[centroid_x, centroid_y] = np.iinfo(np.array(mask_tmp).dtype).max
            #
            # mask = Image.fromarray(mask_tmp)
            # # =================================================

            # synchronized transform
            img, mask = self._sync_transform(img, mask)

            # general resize, normalize and toTensor
            if self.transform is not None:
                img = self.transform(img)
            mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        except Exception as e:
            ex_img_path = 'dataset//NUST-SIRST/images/000000_1.png'
            ex_label_path = 'dataset//NUST-SIRST/masks/000000_1.png'
            img = Image.open(ex_img_path).convert('RGB')
            mask = Image.open(ex_label_path)

            # synchronized transform
            img, mask = self._sync_transform(img, mask)

            # general resize, normalize and toTensor
            if self.transform is not None:
                img = self.transform(img)
            mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
            print(img_path)
            print(label_path)

        return img, torch.from_numpy(mask) #img_id[-1]

    def __len__(self):
        return len(self._items)


class TestSetLoader(Dataset):
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png',label_mode='seg'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix
        self.label_mode = label_mode

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        img_id = self._items[idx]  
        img_path   = self.images+'/'+img_id+self.suffix    
        if self.label_mode == "seg":
            label_path = self.masks + '/' + img_id + self.suffix
        elif self.label_mode == "centroid_mask"  or self.label_mode == "centroid":
            label_path = img_id

        img  = Image.open(img_path).convert('RGB')  
        if self.label_mode == "seg":
            mask = Image.open(label_path)
        elif self.label_mode == "centroid_mask" or self.label_mode == "centroid":
            file_idx = label_path.split("_")[0]
            frame_idx = label_path.split("_")[1]

            x = []
            y = []
            with open(os.path.join(self.masks, "data" + file_idx + ".txt"), "r") as label_file:
                for i, line in enumerate(label_file):
                    if i == int(frame_idx) + 1:
                        line_split = line.split()
                        for j in range(2, len(line_split)):
                            if (j - 2) % 3 == 1:
                                x.append(int(line_split[j]))
                            if (j - 2) % 3 == 2:
                                y.append(int(line_split[j]))
                        break

            mask = np.zeros(img.size).astype('uint8').transpose()
            for i in range(len(x)):
                if self.label_mode == "centroid_mask":
                    x_range = [max(x[i] - 1, 0), min(x[i] + 1, img.width) + 1]
                    y_range = [max(y[i] - 1, 0), min(y[i] + 1, img.height) + 1]
                    mask[y_range[0]:y_range[1], x_range[0]:x_range[1]] = 255
                elif self.label_mode == "centroid":
                    mask[y[i], x[i]] = 255
            mask = Image.fromarray(mask)


        # # =============== test point label ================
        # mask_tmp = np.zeros(mask.size).astype('uint8').transpose()
        #
        # from skimage import measure
        # label = measure.label(np.array(mask), connectivity=2)
        # coord_label = measure.regionprops(label)
        # for i in range(len(coord_label)):
        #     centroid_label = np.array(list(coord_label[i].centroid))
        #     centroid_x = np.rint(centroid_label[0]).astype('int64')
        #     centroid_y = np.rint(centroid_label[1]).astype('int64')
        #     mask_tmp[centroid_x, centroid_y] = np.iinfo(np.array(mask_tmp).dtype).max
        #
        # mask = Image.fromarray(mask_tmp)
        # # =================================================


        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)

class DemoLoader (Dataset):
    
    NUM_CLASS = 1

    def __init__(self, dataset_dir, transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(DemoLoader, self).__init__()
        self.transform = transform
        self.images    = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _demo_sync_transform(self, img):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BILINEAR)

        # final transform
        img = np.array(img)
        return img

    def img_preprocess(self):
        img_path   =  self.images
        img  = Image.open(img_path).convert('RGB')

        # synchronized transform
        img  = self._demo_sync_transform(img)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img



def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        try:
            init.xavier_normal_(m.weight.data)
        except Exception as e:
            print(e)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_ckpt(state, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))

def save_train_log(args, save_dir):
    dict_args=vars(args)
    args_key=list(dict_args.keys())
    args_value = list(dict_args.values())
    with open('result/%s/train_log.txt'%save_dir ,'w') as  f:
        now = datetime.now()
        f.write("time:--")
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write(dt_string)
        f.write('\n')
        for i in range(len(args_key)):
            f.write(args_key[i])
            f.write(':--')
            f.write(str(args_value[i]))
            f.write('\n')
    return

def save_model_and_result(dt_string, epoch,train_loss, test_loss, best_iou, recall, precision, save_mIoU_dir, save_other_metric_dir):

    with open(save_mIoU_dir, 'a') as f:
        f.write('{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t mIoU {:.4f}\n' .format(dt_string, epoch,train_loss, test_loss, best_iou))
    with open(save_other_metric_dir, 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epoch))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')

def save_model(mean_IOU, best_iou, save_dir, save_prefix, train_loss, test_loss, recall, precision, epoch, net):
    if mean_IOU > best_iou:
        save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')
    elif mean_IOU == best_iou and epoch % 10 == 0:
        save_mIoU_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_IoU.log'
        save_other_metric_dir = 'result/' + save_dir + '/' + save_prefix + '_best_IoU_other_metric.log'
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        best_iou = mean_IOU
        save_model_and_result(dt_string, epoch, train_loss, test_loss, best_iou,
                              recall, precision, save_mIoU_dir, save_other_metric_dir)
        save_ckpt({
            'epoch': epoch,
            'state_dict': net,
            'loss': test_loss,
            'mean_IOU': mean_IOU,
        }, save_path='result/' + save_dir,
            filename='mIoU_' + '_' + save_prefix + '_epoch' + '.pth.tar')
    return best_iou

def save_result_for_test(dataset_dir, st_model, epochs, best_iou, recall, precision ):
    with open(dataset_dir + '/' + 'value_result'+'/' + st_model +'_best_IoU.log', 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        f.write('{} - {:04d}:\t{:.4f}\n'.format(dt_string, epochs, best_iou))

    with open(dataset_dir + '/' +'value_result'+'/'+ st_model + '_best_other_metric.log', 'a') as f:
        f.write(dt_string)
        f.write('-')
        f.write(str(epochs))
        f.write('\n')
        f.write('Recall-----:')
        for i in range(len(recall)):
            f.write('   ')
            f.write(str(round(recall[i], 8)))
            f.write('   ')
        f.write('\n')

        f.write('Precision--:')
        for i in range(len(precision)):
            f.write('   ')
            f.write(str(round(precision[i], 8)))
            f.write('   ')
        f.write('\n')
    return

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def make_dir(deep_supervision, dataset, model):
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H:%M:%S")
    # if deep_supervision:
    #     save_dir = "%s_%s_%s_wDS" % (dataset, model, dt_string)
    # else:
    #     save_dir = "%s_%s_%s_woDS" % (dataset, model, dt_string)
    save_dir = "%s_%s_%s" % (dataset, model, dt_string)
    os.makedirs('result/%s' % save_dir, exist_ok=True)
    return save_dir

def total_visulization_generation(dataset_dir, mode, test_txt, suffix, target_image_path, target_dir):
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')



def make_visulization_dir(target_image_path, target_dir):
    if os.path.exists(target_image_path):
        shutil.rmtree(target_image_path)  # recursively delete directory
    os.mkdir(target_image_path)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)  # recursively delete directory
    os.mkdir(target_dir)

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix, base_size):

    # predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.array((torch.sigmoid(pred) > 0.5).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    batch_size = predsss.shape[0]
    for i in range(batch_size):
        img = Image.fromarray(predsss[i,:].reshape(base_size, base_size))
        img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
        img = Image.fromarray(labelsss[i,:].reshape(base_size, base_size))
        img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)


def save_Pred_GT_visulize(pred, img_demo_dir, img_demo_index, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    img = plt.imread(img_demo_dir + '/' + img_demo_index + suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Raw Imamge", size=11)

    plt.subplot(1, 2, 2)
    img = plt.imread(img_demo_dir + '/' + '%s_Pred' % (img_demo_index) +suffix)
    plt.imshow(img, cmap='gray')
    plt.xlabel("Predicts", size=11)


    plt.savefig(img_demo_dir + '/' + img_demo_index + "_fuse" + suffix, facecolor='w', edgecolor='red')
    plt.show()



def save_and_visulize_demo(pred, labels, target_image_path, val_img_ids, num, suffix):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

    return


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# set seed
def seed_torch(seed=1029):
    random.seed(1029)
    os.environ['PYTHONHASHSEED'] = str(1029) # prohibit random hash
    np.random.seed(1029)
    torch.manual_seed(1029)
    torch.cuda.manual_seed(1029)
    torch.cuda.manual_seed_all(1029) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

# recreate directory
def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
