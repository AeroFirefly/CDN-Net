import os.path

import pandas as pd
from scipy.spatial import KDTree
from skimage import measure
from tqdm import tqdm
import cv2
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    '''
    Evaluate detection metrics.
    '''
    thres = 1.5     # detection distance thershold

    df_pred = pd.read_table('hia.IPD', sep='\s+', header=None, encoding='utf-8',
                            names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

    # gt
    fname = os.path.join('../dataset/fits_softCircle_modified/ipdSNR/20200821152001999_7824_824122_LVT04.IPD')
    df_gt = pd.read_table(fname, sep='\s+', header=None, encoding='utf-8',
                          names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar', 'snr'])

    # static snr
    bins_label = [0, 3, 5, 10]
    bins_len = len(bins_label)
    tp_snr_list = []
    fn_snr_list = []
    tp_count_array = np.zeros(bins_len)
    fn_count_array = np.zeros(bins_len)

    # calculate tp, fp, fn

    ''' evaluate by KD-Tree '''
    kd_gt = KDTree(df_gt[['col', 'row']].values)
    match = kd_gt.query(df_pred[['col', 'row']].values, k=1)
    tp_idx_fromgt = np.unique(match[1][match[0] < thres])
    tp_objs = df_gt.copy().iloc[tp_idx_fromgt]
    fn_objs = df_gt.copy().drop(labels=tp_idx_fromgt)
    all_pred = len(df_pred)
    all_gt = len(df_gt)

    for _, tp_item in tp_objs.iterrows():
        snr_item = tp_item['snr']
        if snr_item < 0:
            print('a minus snr!')
        elif snr_item >= bins_label[-1]:
            tp_count_array[-1] += 1
        else:
            for snr_i in range(bins_len - 1):
                if bins_label[snr_i] <= snr_item < bins_label[snr_i + 1]:
                    tp_count_array[snr_i] += 1
                    break
    for _, fn_item in fn_objs.iterrows():
        snr_item = fn_item['snr']
        if snr_item < 0:
            print('a minus snr!')
        elif snr_item >= bins_label[-1]:
            fn_count_array[-1] += 1
        else:
            for snr_i in range(bins_len - 1):
                if bins_label[snr_i] <= snr_item < bins_label[snr_i + 1]:
                    fn_count_array[snr_i] += 1
                    break

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
    print('precision:{} recall:{} f1:{} falseAlarm:{}'.format(precision, recall, f1, faRate))
