from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from wsipack.utils.asap_links import make_asap_link
from wsipack.utils.cool_utils import is_list, is_dict, is_ndarray
from wsipack.wsi.anno_utils import AsapAnno
from .contour_utils import cmp_shape_spacing_factor, px_to_mm2


def get_simple_anno_color_map():
    color_map = {'b':[0,0,200], 'blue':[0,0,200], 'g':[0,200,0], 'orange':[255,165,0],
                 'green':[0,200,0], 'r':[200,0,0], 'red':[200,0,0], None:None}
    return color_map


def write_anno_for_contours(contours, anno_path=None, wsi_path=None, names=None, contour_spacing=None, anno_colors=None,
                            swap_xy=False, compact=True, add_area_to_contour_name=True,
                            min_area=0, wsi_spacing=None, overwrite=False):
    """ contours and names: either lists of contours and their names or a map with group name as key """
    spacing_factor = 1
    if wsi_spacing is not None and contour_spacing is not None:
        spacing_factor = contour_spacing / wsi_spacing
    elif wsi_path is not None:
        # wsi_shape, spacing_factor = cmp_shape_spacing_factor(wsi_path, contour_spacing)
        wsi_shape, spacing_factor = cmp_shape_spacing_factor(wsi_path, contour_spacing)

    color_map = get_simple_anno_color_map()
    colors = ['b','g','r']+[None]*1000

    if is_list(contours):
        contours = {None:contours}
        names = {None:names}
        if anno_colors is not None:
            anno_colors = {None:anno_colors}
    else:
        if anno_colors is not None and not is_dict(anno_colors):
            anno_colors = {k: anno_colors for k in contours.keys()}

    if names is None:
        names = {k:None for k in contours.keys()}
    # for k in list(contours.keys()):
    #     cnts = contours[k]
    # if len(cnts)>0 and cnts[0].dtype==np.uint: #auch int?
    #     cnts = [cnt.astype(np.float) for cnt in cnts]
    #     contours[k] = cnts

    group_contour_areas = defaultdict(list)
    if min_area>0 or add_area_to_contour_name:#opencv only
        for group, cnts in contours.items():

            areas = []
            for c,cnt in enumerate(cnts):
                cnt_area = cv.contourArea(cnt)
                if contour_spacing is not None:
                    cnt_area = px_to_mm2(cnt_area, contour_spacing)
                areas.append(cnt_area)
            group_contour_areas[group] = areas

            if add_area_to_contour_name:
                if names[group] is None:
                    names[group] = [''] * len(cnts)
                final_names = []
                for c,cnt_area in enumerate(areas):
                    fomrat = '_area_%.3f'
                    if cnt_area < 1e-3:
                        fomrat = '_area_%.2e'
                    final_names.append(names[group][c] + fomrat % cnt_area)
                names[group] = final_names

    if anno_colors is None:
        anno_colors = {}
        for i,key in enumerate(sorted(contours.keys())):
            anno_colors[key] = colors[i]
    group_keys = list(anno_colors.keys())
    for group_key in group_keys:
        group_col = anno_colors[group_key]
        if group_col in color_map.keys():
            anno_colors[group_key] = color_map[group_col]

    annotation = AsapAnno()

    for group, group_color in anno_colors.items():
        if group is not None:
            annotation.add_group(group, group_color)

    skipped = 0;
    for group,cnts in contours.items():
        cnt_areas = group_contour_areas[group]
        for c,contour in enumerate(cnts):
            if min_area > 0 and cnt_areas[c] < min_area:
                skipped+=1
                continue
            if compact and len(contour)>10:
                # epsilon = 0.01 * cv.arcLength(cnt, True)
                epsilon = 1
                contour = cv.approxPolyDP(contour, epsilon, True)
            contour = contour.squeeze()
            name = None
            if names is not None and names[group] is not None and len(names[group])>0:
                name = names[group][c]

            if spacing_factor != 1:
                contour = contour * spacing_factor
            if swap_xy:
                if len(contour.shape)==1 and contour.shape[0]==2:
                    contour = contour[[1,0]]
                else:
                    contour = contour[:, [1, 0]]

            if contour.shape[0] >= 3:
                # todo check for rects 'rectangle'
                target_type = 'polygon'
                annotation.add_polygon(name, group, coords=contour.astype(float), color=anno_colors[group])
                # annotation.add(annotation=target_type, coordinates=contour.astype(np.float), name=name,
                #                group=group, color=anno_colors[group])
            else:
                skipped += 1
    if anno_path is not None:
        annotation.save(str(anno_path), overwrite=overwrite)
    # print('%s saved, skipped %d areas' % (Path(anno_path).name, skipped))
    print('skipped %d areas' % (skipped))
    return annotation
