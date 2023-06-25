from pathlib import Path

import skimage.draw
import skimage.morphology

from wsipack.utils.cool_utils import is_list, is_ndarray, is_tuple
import cv2 as cv
import numpy as np
from scipy.spatial.distance import cdist

from wsipack.wsi.wsd_image import ImageReader

def find_contours(arr, only_outer=True, convex=False):
    if only_outer:
        mode = cv.RETR_EXTERNAL
    else:
        mode = cv.RETR_LIST
    cresults = cv.findContours(arr.astype(np.uint8), mode, cv.CHAIN_APPROX_SIMPLE)
    cresults = list(cresults)
    if len(cresults)==3: #where the contours are depends on cv version
        contours=cresults[1]
    else:
        contours = cresults[0]
    if is_tuple(contours):
        contours = list(contours)

    if convex:
        contours = [cv.convexHull(cnt) for cnt in contours]
    return contours


def contour_distance(cnt1, cnt2):
    """ returns the minimum distance between contours (between the closest edges) """
    if len(cnt1.shape) == 3: #for some reason like this with empty middle dimension (n,1,2)
        cnt1 = np.squeeze(cnt1)
        cnt2 = np.squeeze(cnt2)

    dists = cdist(cnt1, cnt2, 'euclidean')
    dist_mins = dists.min()
    return dist_mins

def contour_distances(cnts):
    """ computes the inter-contour distances """
    dists = np.zeros((len(cnts), len(cnts)))
    for i,cnti in enumerate(cnts):
        for j in range(i+1, len(cnts)):
            cntj = cnts[j]
            dist = contour_distance(cnti, cntj)
            dists[i,j] = dist
            dists[j,i] = dist
    return dists

def contour_areas(contours, spacing):
    areas = [cv.contourArea(cnt) for cnt in contours]
    areas = [px_to_mm2(a, spacing) for a in areas]
    return areas

def cmp_shape_spacing_factor(wsi_path, spacing):
    # if isinstance(wsi_path, ImageReader):
    #     wsi = wsi_path
    #     close_reader = False
    if isinstance(wsi_path, (str, Path)):
        wsi = ImageReader(str(wsi_path))
        close_reader = True
    else:
        wsi = wsi_path
        close_reader = False
    wsi_spacing = wsi.spacings[0]
    wsi_level = wsi.level(spacing)
    wsi_shape = wsi.shapes[wsi_level]
    if close_reader:
        wsi.close()
    spacing = wsi.refine(spacing)
    spacing_factor = spacing / wsi_spacing
    return wsi_shape, spacing_factor
    #example: wsi_spacing=0.5, mask_spacing=2.0, factor=4
    #example: wsi_spacing=2.0, mask_spacing=4.0, factor=2

def contours_wsi_adjust(wsi_path, contours, contour_spacing):
    """ adjusts the contours computed at contour_spacing for the wsi """
    wsi_shape, spacing_factor = cmp_shape_spacing_factor(wsi_path, contour_spacing)
    if spacing_factor != 1:
        contours = [contour * spacing_factor for contour in contours]  # not at wsi-spacing
    return contours


def dilate_binary_image_skimage(bin_img, distance = 1):
    if distance<=1:
        neighborhood = None
    else:
        neighborhood = np.zeros((distance*2+1, distance*2+1), dtype=np.bool)
        rr, cc = skimage.draw.ellipse(distance, distance, distance+0.5, distance+0.5)
        neighborhood[rr, cc] = 1
        # showim(neighborhood)

    return skimage.morphology.binary_dilation(image = bin_img, selem=neighborhood)

def _create_ellipse_kernel(distance):
    if distance<=1:
        neighborhood = None
    else:
        neighborhood = np.zeros((distance*2+1, distance*2+1), dtype=np.uint8)
        rr, cc = skimage.draw.ellipse(distance, distance, distance+0.5, distance+0.5)
        neighborhood[rr, cc] = 1
        # showim(neighborhood)
    return neighborhood

def dilate_image(img, distance=1, **kwargs):
    kernel = _create_ellipse_kernel(distance)
    return cv.dilate(img, kernel=kernel, **kwargs)

def erode_image(img, distance=1, **kwargs):
    kernel = _create_ellipse_kernel(distance)
    return cv.erode(img, kernel=kernel, **kwargs)

def open_image(img, distance=1, **kwargs):
    kernel = _create_ellipse_kernel(distance)
    return cv.morphologyEx(img, cv.MORPH_OPEN, kernel, **kwargs)

def grayscale_to_rgb(img):
    if len(img.shape)==2:
        img = np.expand_dims(img, 2)
    if img.shape[2]==1:
        img = np.repeat(img, 3, axis=-1)
    return img


def contours_to_mask_skimage(contours, shape):
    mask = np.zeros(shape[:2], dtype='bool')
    for contour in contours:
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        # mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1

        rr, cc = skimage.draw.polygon(np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int'), mask.shape)
        mask[rr, cc] = 1

        # showim(mask)
        # mask_c = ndimage.binary_fill_holes(mask_c)
        # mask+=mask_c
    # r_mask = ~r_mask
    return mask

def contours_as_mask(contours, shape, val=255, rgb=False):
    if is_ndarray(contours):#single contour
        contours = [contours] #otherwise only the points will be drawn
    if rgb:
        mask = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
        color = (val, val, val)

    else:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        color=(val)
    cv.fillPoly(mask, pts=contours, color=color)
    return mask

def contours_to_mask(contours, mask, val=255, xy=True):
    if len(mask.shape)==3 and mask.shape[2]==3:
        color = (val, val, val)
    else:
        color = (val)
    if is_list(contours):
        contours = np.array(contours, dtype=np.int32)
    if is_ndarray(contours):
        if contours.dtype!=np.int32:
            contours = contours.astype(np.int32)
        if len(contours.shape)==2:
            contours = contours[None,:]
    if not xy:
        contours = contours[:,:,[1, 0]].copy()
    cv.fillPoly(mask, pts=contours, color=color)

def contours_draw(contours, shape, val=255, rgb=False):
    if rgb:
        mask = np.zeros((shape[0],shape[1],3), dtype=np.uint8)
        color = (val, val, val)

    else:
        mask = np.zeros(shape[:2], dtype=np.uint8)  # create a single channel 200x200 pixel black image
        color=(val)
    cv.drawContours(mask, contours, color=color)
    return mask

def contours_draw_to(contours, img, val=255):
    if len(img.shape)==3 and img.shape[2]==3:
        color = (val, val, val)
    else:
        color = (val)
    cv.fillPoly(img, pts=contours, color=color)

def contour_minwidth(cnt, shape=None):
    rect = cv.minAreaRect(cnt)
    (x, y), (width, height), angle = rect
    min_width = min(width, height)
    if shape is not None: #for debug
        box = cv.boxPoints(rect)
        box = np.int0(box)
        img = np.zeros(shape[:2], dtype=np.uint8)
        cv.drawContours(img, [cnt], 0, (255), 2)
        cv.drawContours(img, [box], 0, (255), 2)
        # showim(img)
    return min_width

def dist_to_px(dist, spacing):
    """ distance in um (or rather same unit as the spacing) """
    dist_px = int(round(dist / spacing))
    return dist_px

def px_to_dist(px, spacing):
    dist = int(round(px * spacing))
    return dist

def px_to_mm2(px, spacing):
    # spacing unit: um/px
    area_um2 = px*(spacing**2) #ideally spacing_vertical*spacing_horzontal
    area_mm2 = area_um2/1e6
    return area_mm2

def mm2_to_px(area, spacing):
    # spacing unit: um/px
    px = area*1e6/(spacing**2)
    return px



def df_px_to_mm2(df, tissue_cols, spacing=2.0):
    """ translates roughly the px amount to um**2 """
    ignore_parts = ['tils']
    failed_cols = []
    for col in tissue_cols:
        if col in df:
            skip_col = False
            for ip in ignore_parts:
                if ip in col:
                    skip_col = True
                    break
            if skip_col:
                continue
            try:
                df[col] = px_to_mm2(df[col], spacing=spacing)
            except:
                failed_cols.append(col)
    if len(failed_cols)>0:
        print('mm2 didnt work for %s' % str(failed_cols))
    return failed_cols

def _test_pixel_to_mm2():
    a1 = px_to_mm2(1000000, spacing=1)
    assert np.isclose(a1,1)
    a1 = px_to_mm2(4 * 1000000, spacing=0.5)
    assert np.isclose(a1,1)

if __name__ == '__main__':
    _test_pixel_to_mm2()

    # path = "<pred_tif>"
    # reader = ImageReader(str(path))
    # content = reader.content(reader.spacings[-1]).squeeze()
    # # print(content.shape)
    # bcont = content>0
    # cnt = find_contours(bcont)
    # # ellipse = cv.fitEllipse(cnt)
    # contour_minwidth(cnt[1], content.shape)

    # mask = contours_to_mask_skimage(cnt, bcont.shape)
    # mask = contours_as_mask(cnt, bcont.shape)
    # eroded = erode_image(mask, distance=2)
    # # print(mask.shape, eroded.shape)
    # showims(mask, eroded)
    #
    # cnt_eroded = find_contours(eroded)
    #
    # cnt_img = contours_as_mask(cnt, bcont.shape)
    # # contours_to_mask(cnt_eroded, cnt_img)
    # showim(cnt_img)

    # bcont_img = bcont.astype(np.uint8)
    # bcont_img = grayscale_to_rgb(bcont_img)
    # print(bcont_img.shape)
    # cv.drawContours(bcont_img, cnt, -1, (0, 255, 0), 3)
    # cv.ellipse(img, ellipse, (0, 255, 0), 2)
    # showim(bcont_img)

