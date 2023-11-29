from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    
    # choose model
    parser.add_argument('--model', type=str, default='cdnnet', help='CDNNet, DNANet, StarNet')
    
    # parameter for CDNNet
    parser.add_argument('--channel_size', type=str, default='three', help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_18', help='vgg10, resnet_10, resnet_18, resnet_34')
    parser.add_argument('--netdepth', type=int, default='4', help='3, 4, 5')
    parser.add_argument('--scale_method', type=str, default='deconv', help='deconv')
    parser.add_argument('--deep_supervision', type=str, default='True', help='True or False')
    
    # data and pre-process
    parser.add_argument('--dataset', type=str, default='fits_softCircle_modified', help='dataset name: fits_softCircle_modified, fits_starSimu')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--label_mode', type=str, default='seg', help='label mode: seg, centroid_mask')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/', help='dataset root directory')
    parser.add_argument('--suffix', type=str, default='.png', help='seg label suffix')
    parser.add_argument('--split_method', type=str, default='2304_256', help='2304_256 for both real and simu dataset')
    parser.add_argument('--preprocess', type=str, default='hierarchy', help='hierarchy')
    parser.add_argument('--workers', type=int, default=2, metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3, help='default 3')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')

    #  hyper params for testing
    parser.add_argument('--st_model', type=str, default='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS', help='model store directory')
    parser.add_argument('--model_dir', type=str, default = 'NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar', help='model path')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='input batch size for testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0', help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10, help='crop image size')

    ## parse args
    args = parser.parse_args()
    if args.preprocess.lower()=='hierarchy' and args.in_channels!=3:
        args.in_channels = 3
        print('in_channels changed to 3')
    return args