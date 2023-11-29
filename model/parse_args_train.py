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
    parser.add_argument('--test_size', type=float, default='0.5', help='when mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/', help='dataset root directory')
    parser.add_argument('--suffix', type=str, default='.png', help='seg label suffix')
    parser.add_argument('--split_method', type=str, default='2304_256', help='2304_256 for both real and simu dataset')
    parser.add_argument('--preprocess', type=str, default='hierarchy', help='hierarchy')
    parser.add_argument('--mask_type', type=str, default='soft', help='soft')
    parser.add_argument('--workers', type=int, default=2, metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3, help='default 3')
    parser.add_argument('--base_size', type=int, default=256, help='base image size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop image size')

    # hyper params for training
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=4, metavar='N', help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='N', help='input batch size for testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help=' Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--loss_func', type=str, default='SIoU+SFL', help='SIoU+SFL')
    
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0', help='Training with GPUs, you can specify 1,3 for example.')
    
    # resume training
    parser.add_argument('--resume_from', type=str, default='', help='resume training from this path')

    ## parse args
    args = parser.parse_args()
    if args.preprocess.lower()=='hierarchy' and args.in_channels!=3:
        args.in_channels = 3
        print('in_channels is set to 3, under hierarchy preprocess training mode')
    args.save_dir = make_dir(args.deep_supervision, args.dataset, args.model) # make dir for save result
    save_train_log(args, args.save_dir) # save training log
    return args