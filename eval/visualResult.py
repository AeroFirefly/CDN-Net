"""
  @Author  : Xinyang Li
  @Time    : 2022/5/17 下午4:49
"""
import os

import numpy as np
from astropy.io import fits
import shutil
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import KDTree
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import cv2


def recreate_dir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

def visualDet(dets,
               img_name,
               visual_dir=None,
               wh=[4096, 4096],
               png=None):
    if visual_dir is not None:
        draw = ImageDraw.ImageDraw(png)
        visual_det = os.path.join(visual_dir, img_name + '.png')

    for det_id, det_item in enumerate(dets):
        cen_x, cen_y, area, width, height = list(map(float, det_item.split()[:5]))

        width_int = int(round(width))
        height_int = int(round(height))

        st_x, st_y, end_x, end_y = [cen_x - width / 2, cen_y - height / 2,
                                    cen_x + width / 2, cen_y + height / 2]
        st_x_int = max(0, int(round(st_x))-1)
        st_y_int = max(0, int(round(st_y))-1)
        end_x_int = min(wh[0] - 1, int(round(end_x))-1)
        end_y_int = min(wh[1] - 1, int(round(end_y))-1)

        try:
            if visual_dir is not None:
                draw.rectangle(((st_x_int, st_y_int),
                                (end_x_int, end_y_int)), fill=None, outline='red', width=1)

        except Exception as e:
            print('draw mask error: [{0}, {1}, {2}, {3}]'.format(st_x_int,
                                                                 st_y_int,
                                                                 end_x_int,
                                                                 end_y_int))
    if visual_dir is not None:
        png.save(visual_det)


if __name__ == '__main__':
    method_res = 'cdn_result'
    res_dir = os.path.join('../dataset/fits_softCircle_modified', method_res)
    out_png_dir = os.path.join(res_dir, 'png')
    recreate_dir(out_png_dir)
    out_bbox_dir = os.path.join(res_dir, 'bbox')
    recreate_dir(out_bbox_dir)

    # # 1. png-whole
    fname = '20200821152001999_7824_824122_LVT04'
    img_fits = fits.getdata('../fitstrans/fitsdata/4/20200821152001999_7824_824122_LVT04.fits')
    img_fits = img_fits.clip(img_fits.mean()-img_fits.std(), img_fits.mean()+3*img_fits.std())
    Image.fromarray(img_fits).convert('L').save(os.path.join(res_dir, '{}_pngwhole.png'.format(fname)))

    # 2. bbox-whole
    im_gray = np.array(Image.open(os.path.join(res_dir, '{}_pngwhole.png'.format(fname))))
    img_color = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
    if method_res=='sex_result':
        df_pred = pd.read_table('../dataset/fits_softCircle_modified/{}/20200821152001999_7824_824122_LVT04.cat'.format(method_res),
                                sep='\s+', header=None, skiprows=17, encoding='utf-8',
                                names=['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'XWIN_IMAGE', 'YWIN_IMAGE', 'col', 'row',
                                       'MAG_AUTO',
                                       'MAG_MODEL', 'FWHM_IMAGE', 'FLUX_RADIUS', 'FLAGS', 'FLAGS_WIN', 'NITER_WIN',
                                       'SPREAD_MODEL', 'SPREADERR_MODEL', 'ID_PARENT'])
    else:
        df_pred = pd.read_table('../dataset/fits_softCircle_modified/{}/20200821152001999_7824_824122_LVT04.IPD'.format(method_res),
                                sep='\s+', header=None, encoding='utf-8',
                                names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])
    df_gt = pd.read_table('../dataset/fits_softCircle_modified/ipd/20200821152001999_7824_824122_LVT04.IPD',
                          sep='\s+', header=None, encoding='utf-8',
                          names=['col', 'row', 'pixnum', 'length', 'width', 'graysum', 'bgavg', 'bgvar'])

    ''' evaluate by KD-Tree '''
    thres = 1.5
    all_pred = len(df_pred)
    all_gt = len(df_gt)

    kd_gt = KDTree(df_gt[['col', 'row']].values)
    match = kd_gt.query(df_pred[['col', 'row']].values, k=1)
    tp_idx_fromgt = np.unique(match[1][match[0] < thres])
    tp_objs_all = df_gt.copy().iloc[tp_idx_fromgt]
    fn_objs_all = df_gt.copy().drop(labels=tp_idx_fromgt)

    kd_pred = KDTree(df_pred[['col', 'row']].values)
    match = kd_pred.query(df_gt[['col', 'row']].values, k=1)
    tp_idx_frompred = np.unique(match[1][match[0] < thres])
    fp_objs_all = df_pred.copy().drop(labels=tp_idx_frompred)

    region_x = [512, 768, 1024, 1280]
    region_y = [768, 256, 2048, 512]
    region_x = list(map(lambda x: int(x/256), region_x))
    region_y = list(map(lambda x: int(x/256), region_y))


    for i in range(len(region_x)):
            try:
                x_st = 256 * region_x[i]
                x_end = 256 * (region_x[i]+1)
                y_st = 256 * region_y[i]
                y_end = 256 * (region_y[i]+1)
                img_color_patch = img_color[y_st:y_end, x_st:x_end, :]
                tp_objs = tp_objs_all.copy()[(tp_objs_all['row'] >= y_st) & (tp_objs_all['row'] < y_end)
                                             & (tp_objs_all['col'] >= x_st) & (tp_objs_all['col'] < x_end)]
                fn_objs = fn_objs_all.copy()[(fn_objs_all['row'] >= y_st) & (fn_objs_all['row'] < y_end)
                                             & (fn_objs_all['col'] >= x_st) & (fn_objs_all['col'] < x_end)]
                fp_objs = fp_objs_all.copy()[(fp_objs_all['row'] >= y_st) & (fp_objs_all['row'] < y_end)
                                             & (fp_objs_all['col'] >= x_st) & (fp_objs_all['col'] < x_end)]

                ''' plot an ellipse for each object '''
                fig, ax = plt.subplots()
                for i in range(len(tp_objs)):
                    ab_mean = (tp_objs.iloc[i].loc['width'] + tp_objs.iloc[i].loc['length']) / 2
                    e = Ellipse(xy=(tp_objs.iloc[i].loc['col']-x_st, tp_objs.iloc[i].loc['row']-y_st),
                                width=ab_mean,
                                height=ab_mean,
                                angle=0)
                    e.set_facecolor('none')
                    e.set_edgecolor('green')
                    ax.add_artist(e)
                for j in range(len(fn_objs)):
                    ab_mean = (fn_objs.iloc[j].loc['width'] + fn_objs.iloc[j].loc['length']) / 2
                    e = Ellipse(xy=(fn_objs.iloc[j].loc['col']-x_st, fn_objs.iloc[j].loc['row']-y_st),
                                width=ab_mean,
                                height=ab_mean,
                                angle=0)
                    e.set_facecolor('none')
                    e.set_edgecolor('red')
                    ax.add_artist(e)
                for k in range(len(fp_objs)):
                    if method_res == 'sex_result':
                        ab_mean = fp_objs.iloc[k].loc['FLUX_RADIUS'] * 2
                    else:
                        ab_mean = (fp_objs.iloc[k].loc['width'] + fp_objs.iloc[k].loc['length']) / 2
                    e = Ellipse(xy=(fp_objs.iloc[k].loc['col']-x_st, fp_objs.iloc[k].loc['row']-y_st),
                                width=ab_mean,
                                height=ab_mean,
                                angle=0)
                    e.set_facecolor('none')
                    e.set_edgecolor('orange')
                    ax.add_artist(e)
                im = ax.imshow(img_color_patch)
                plt.axis('off')
                plt.savefig(os.path.join(out_bbox_dir, '{}-{}-{}-256.png'.format(fname,str(x_st),str(y_st))), bbox_inches='tight', pad_inches=0.)
                # plt.show()
            except Exception as e:
                print('aa')
