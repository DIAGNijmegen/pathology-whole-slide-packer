from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from wsipack.utils.asap_links import make_asap_link
from wsipack.utils.cool_utils import ensure_dir_exists, is_iterable, take_closest_number
from wsipack.utils.flexparse import FlexArgumentParser
from wsipack.utils.path_utils import PathUtils, get_path_named_like
from wsipack.wsi.anno_utils import AsapAnno
from wsipack.wsi.wsd_image import ImageReader

allowed_spacings = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def _get_slide_spacing(path):
    reader = ImageReader(str(path))
    spacing = reader.spacings[0]
    reader.close()
    return spacing


def pack_annos(annos_dir, packed_images_dir, wsi_dir_or_pathes, out_dir=None, out_dir_prefix=None, overwrite=False, nolinks=False):
    print('pack annos: annos_dir=%s, packed_images_dir=%s' % (annos_dir, packed_images_dir))
    # print('annotation names must match the slide name')
    anno_dir_name = Path(annos_dir).name if out_dir_prefix is None else out_dir_prefix
    packed_images_dir = Path(packed_images_dir)
    packed_dir = packed_images_dir.parent
    if out_dir is None:
        out_dir = packed_dir
    out_dir = Path(out_dir)
    print('anno out_dir: %s' % str(out_dir))
    out_dir_packed_annos = out_dir/(anno_dir_name+'_packed')
    out_dir_packed_annos_links = str(out_dir_packed_annos)+'_links'
    out_dir_failed_annos = out_dir/(anno_dir_name+'_failed')
    out_dir_failed_annos_links = str(out_dir_failed_annos)+'_links'

    ensure_dir_exists(out_dir_packed_annos)

    anno_pathes = PathUtils.list_pathes(annos_dir, ending='xml')
    if is_iterable(wsi_dir_or_pathes) and Path(wsi_dir_or_pathes[0]).exists():
        wsi_pathes = wsi_dir_or_pathes
    else:
        print('listing wsis in %s' % wsi_dir_or_pathes)
        wsi_pathes = PathUtils.list_pathes(wsi_dir_or_pathes, containing_or=['mrxs', 'tif', 'svs', 'ndpi'], type='file')

    print('found %d wsis' % len(wsi_pathes))

    packed_anno_dir = Path(packed_dir)/'tissue_anno'
    # packed_anno_dir = Path(packed_dir)/'images_packed_anno' #obsolete
    orig_anno_dir = Path(packed_dir)/'orig_tissue_anno'
    # orig_anno_dir = Path(packed_dir)/'images_orig_anno' #obsolete

    packed_roi_anno_pathes = PathUtils.list_pathes(packed_anno_dir, ending='xml')

    #For each packed slide take the roi-annos containing the rois for each slide (group name)
    print('processing %s roi annos' % len(packed_roi_anno_pathes))
    for packed_roi_anno_path in tqdm(packed_roi_anno_pathes):
        # packed_roi_anno = Annotation()
        # packed_roi_anno.open(str(packed_roi_anno_path))
        packed_roi_anno = AsapAnno(packed_roi_anno_path)

        packed_path = packed_images_dir/(packed_roi_anno_path.stem+'.tif')
        if not packed_path.exists(): raise ValueError('couldnt find packed %s' % packed_path)
        packed_spacing = _get_slide_spacing(packed_path)
        #FIXME: check why allowed_spacings is used
        packed_spacing = take_closest_number(allowed_spacings, packed_spacing)
        #Find the anno pathes and the corresponding wsi pathes
        orig_anno_pathes_p = []; wsi_pathes_p = []
        for gr in packed_roi_anno.groups: #groups have the original slide names
            slide_name = gr.name
            anno_path = get_path_named_like(slide_name, anno_pathes, same_name=True)
            if anno_path is not None:
                orig_anno_pathes_p.append(anno_path)
                wsi_path = get_path_named_like(slide_name, wsi_pathes, same_name=True)
                if wsi_path is None:
                    raise ValueError('found anno %s, but not the corresponding wsi' % (anno_path))
                wsi_pathes_p.append(wsi_path)

        #Convert the individual annos to the packed coordinates and also extract annos which could not be packed (transfered)
        print('packing %d annos' % len(orig_anno_pathes_p))
        packed_annos = []; packed_annos_wsi_names = []
        for orig_anno_path, wsi_path in zip(orig_anno_pathes_p, wsi_pathes_p):
            roi_anno_path = orig_anno_dir/orig_anno_path.name
            if not roi_anno_path.exists(): raise ValueError('couldnt find orig roi anno path %s' % orig_anno_path)
            wsi_spacing = _get_slide_spacing(wsi_path)
            wsi_spacing = take_closest_number(allowed_spacings, wsi_spacing)
            packed_anno, bad_orig_anno = _pack_anno(orig_anno_path, roi_anno_path, packed_anno_path=packed_roi_anno_path,
                                               wsi_spacing=wsi_spacing, packed_spacing=packed_spacing)
            packed_annos.append(packed_anno)
            packed_annos_wsi_names.append(Path(wsi_path).stem)

            if bad_orig_anno is not None:
                ensure_dir_exists(out_dir_failed_annos)
                failed_anno_path = Path(out_dir_failed_annos)/orig_anno_path.name
                bad_orig_anno.save(str(failed_anno_path), overwrite=overwrite)
                if not nolinks:
                    make_asap_link(wsi_path, mask_path=None, anno_path=failed_anno_path, links_dir=out_dir_failed_annos_links)

        #Merge packed annos if necessary and write
        if len(packed_annos)==0:
            packed_anno = None
        else:
            packed_anno = packed_annos[0]
            if len(packed_annos)>1:
                #todo: test anno merging!
                for j in range(1, len(packed_annos)):
                    prefix1 = packed_annos_wsi_names[0] if j==1 else ''
                    prefix2 = packed_annos_wsi_names[j]

                    # packed_anno = merge_annotations(packed_anno, packed_annos[j+1],
                    #                                 anno_prefix1=prefix1, anno_prefix2=prefix2)
                    packed_anno = AsapAnno.merge(packed_anno, packed_annos[j],
                                                 anno_prefix1=prefix1, anno_prefix2=prefix2)

            packed_anno_out_path = out_dir_packed_annos/(packed_path.stem+'.xml')
            # packed_anno.save(str(packed_anno_out_path))
            packed_anno.save(packed_anno_out_path, overwrite=overwrite)
            if not nolinks:
                make_asap_link(packed_path, mask_path=None, anno_path=packed_anno_out_path, links_dir=out_dir_packed_annos_links)


def _pack_anno(anno_path, roi_anno_path, packed_anno_path, wsi_spacing, packed_spacing, wsi_name=None):
    if wsi_name is None:
        wsi_name = Path(anno_path).stem
    # packed_path = Path(packed_path)
    # packed_anno_dir = packed_path.parent/'images_packed_anno'
    # packed_anno_path = packed_anno_dir/(packed_path.stem+'.xml')

    spacing_factor = wsi_spacing/packed_spacing
    #read annos etc.
    anno = AsapAnno(anno_path)

    #the info is also in packed_anno in the name, but the overlap code uses annos,
    #so instead of converting the name into rectangle-anno just use the roi_anno
    roi_anno = AsapAnno(roi_anno_path)
    packed_anno = AsapAnno(packed_anno_path)

    packed_roi_annos = packed_anno.get_annos_in_group(wsi_name)

    all_found_annos = []
    for packed_roi in packed_roi_annos:
        orig_roi = roi_anno.get_anno_by_name(packed_roi.name)
        found_annos = anno.get_annos_overlapping_with(orig_roi)
        all_found_annos.extend(found_annos)

        shift0 = orig_roi.coords[0]
        anno.shift_annos(found_annos, -shift0[0], -shift0[1])
        anno.scale_annos(found_annos, spacing_factor)
        shiftXY = packed_roi.coords[0]
        anno.shift_annos(found_annos, shiftXY[0], shiftXY[1])

    # unmatched_annos = [anno_item for anno_item in anno.annotations if anno_item not in all_found_annos]
    unmatched_annos = [anno_item for anno_item in anno.annos if anno_item not in all_found_annos]
    unma_anno = None
    if len(unmatched_annos)>0:
        print('%d/%d annos out of rois!!' % (len(unmatched_annos), len(all_found_annos)))
        unma_anno = deepcopy(anno)
        unma_anno.remove_annos(all_found_annos)
        #remove from the (packed) anno
        anno.remove_annos(unmatched_annos)

    return anno, unma_anno


if __name__ == '__main__':
    parser = FlexArgumentParser()
    parser.add_argument('--anno_dir', type=str, required=True, help='asap annotations directory')
    parser.add_argument('--packed_images_dir', type=str, required=True, default='directory with the packed images')
    parser.add_argument('--wsi_dir_or_pathes', type=str, required=True,
                        default='directory with the original slides, necessary to determine the original spacings')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory, default is the parent of the packed images')
    parser.add_argument('--nolinks', action='store_true', help='dont create asap links')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')

    args = parser.parse_args()
    print('args:', args)
    pack_annos(**args)

