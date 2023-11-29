import os.path

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams["font.family"] = ["Times New Roman"] + plt.rcParams['font.serif']
from matplotlib.pyplot import MultipleLocator
import numpy as np

def getCurveData(fpath, epoch_range):
    curve_pre = []
    curve_rec = []
    with open(fpath) as f:
        while True:
            line = f.readline()
            if line.strip():
                line_split = line.split()
                pre = line_split[1].split(':')[-1].replace(',', '')
                rec = line_split[2].split(':')[-1].replace(',', '')
                curve_pre.append(float(pre))
                curve_rec.append(float(rec))
            else:
                break
    curve_pre = np.array(curve_pre)[epoch_range]
    curve_rec = np.array(curve_rec)[epoch_range]
    return curve_pre, curve_rec

def drawAndSave(fpath_list, epoch_range, label_list, outpath):
    curve_num = len(fpath_list)
    curve_pre_list = [None] * curve_num
    curve_rec_list = [None] * curve_num
    color_list = ['red', 'blue', 'green']
    for i in range(curve_num):
        fpath = fpath_list[i]
        curve_pre_list[i], curve_rec_list[i] = getCurveData(fpath, epoch_range)

    plt.figure()
    # for i in range(curve_num):
    #     if i==0:
    #         plt.plot(epoch_range, curve_pre_list[i], color=color_list[i], label="{} precision".format(label_list[i]))
    #     else:
    #         plt.plot(epoch_range, curve_pre_list[i], color=color_list[i], label="{} precision".format(label_list[i]))
    # # l1 = plt.legend(loc='lower right')
    handle_list = [None] * curve_num
    for i in range(curve_num):
        if i==0:
            handle_list[i], = plt.plot(epoch_range, curve_rec_list[i], color=color_list[i],
                                      label="{}".format(label_list[i]))
        else:
            handle_list[i], = plt.plot(epoch_range, curve_rec_list[i], color=color_list[i], linestyle='dashed',
                                      label="{}".format(label_list[i]))
    # plt.legend(handles=handle_list, loc='upper right', scatterpoints=1)
    plt.legend(loc='lower right', scatterpoints=1)
    plt.xlabel('epoch', fontsize=12)
    plt.ylabel('detection rate', fontsize=12)
    # plt.gca().add_artist(l1)
    # 定义横坐标刻度间隔对象, 间隔为1, 代表每一轮次
    x_major_locator = MultipleLocator(5)
    # 获得当前坐标图句柄
    ax = plt.gca()
    # 设置横坐标刻度间隔
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(epoch_range.min(), epoch_range.max())
    plt.grid()
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':

    ''' 1. wo deconv '''
    method_name = 'deconv'
    fpath_list = [os.path.join(
                      '../result/CDNNet-Ablation-Result/A.LevelNum/level=4/DNANet_fits_softCircle_modified_good_precision_recall.log'
                  ),
                  os.path.join(
                      '../result/CDNNet-Ablation-Result/B.Deconv/B1_interp/DNANet_fits_softCircle_modified_good_precision_recall.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['CDN-Net', 'wo/{}'.format(method_name)]
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)

    ''' 2. wo hierarchy '''
    method_name = 'hierarchy'
    fpath_list = [os.path.join(
                      '../result/CDNNet-Ablation-Result/A.LevelNum/level=4/DNANet_fits_softCircle_modified_good_precision_recall.log'
                  ),
                  os.path.join(
                      '../result/CDNNet-Ablation-Result/C.Hierarchy/C3_sigmacut/DNANet_fits_softCircle_modified_good_precision_recall.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['CDN-Net', 'wo/{}'.format(method_name)]
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)

    ''' 3. wo softmask '''
    method_name = 'softmask'
    fpath_list = [os.path.join(
                      '../result/CDNNet-Ablation-Result/A.LevelNum/level=4/DNANet_fits_softCircle_modified_good_precision_recall.log'
                  ),
                  os.path.join(
                      '../result/CDNNet-Ablation-Result/D.SoftMask/D1_hardMask/DNANet_TZB_fits_modified_good_precision_recall.log'
                  )
                  ]
    epoch_range = np.arange(70, 100)
    label_list = ['CDN-Net', 'wo/{}'.format(method_name)]
    outpath = '../outfiles/plotDetMetricCurve/wo{}.png'.format(method_name)
    drawAndSave(fpath_list, epoch_range, label_list, outpath)
