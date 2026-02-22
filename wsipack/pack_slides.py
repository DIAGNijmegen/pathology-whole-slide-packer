import getpass, traceback, argparse
from copy import deepcopy

import cv2 as cv
import pandas as pd
import rpack
from time import sleep

from tqdm import tqdm
from wsipack.utils.files_utils import create_pathes_csv
from wsipack.utils.flexparse import FlexArgumentParser
from wsipack.wsi.asap_writer import ArrayImageWriter
from wsipack.wsi.resize_masks import resize_mask
#pyvips must be imported before multiresolutionimageinterface otherwise libMagickCore error!
from wsipack.utils.signal_utils import ExitHandler
from wsipack.pack_annos import pack_annos, allowed_spacings

from wsipack.utils.asap_links import make_asap_link
from wsipack.utils.cool_utils import *
from wsipack.utils.path_utils import get_corresponding_pathes, get_path_named_like, get_matching_pathes, PathUtils
from wsipack.wsi.wsi_read import ImageReader, create_reader, InvalidSpacingError
from wsipack.wsi.wsi_utils import create_thumbnail

from wsipack.wsi.contour_utils import find_contours, px_to_mm2, cmp_shape_spacing_factor
from wsipack.wsi.contour_anno_utils import write_anno_for_contours

from wsipack.wsi.slide_summary import create_slides_info_summary

from wsipack.utils.geometry import merge_overlapping_bboxes

print = print_mp


def _rpack_exp():

    #rectangles (width, height)
    sizes = [(58, 206), (231, 176), (35, 113), (46, 109)]
    # sizes = [[58, 206], [231, 176], [35, 113], [46, 109]]
    positions = rpack.pack(sizes)
    bsize = rpack.bbox_size(sizes, positions)

    # The result will be a list of (x, y) positions:
    print(positions)
    print(bsize)
    #[(0, 0), (58, 0), (289, 0), (289, 113)]

def _read_content(mask_path, spacing):
    """ return content and factor. coords0=coords*factor"""
    reader = create_reader(str(mask_path))
    content = reader.content(spacing)
    spacing = reader.refine(spacing)
    factor = spacing/reader.spacings[0]
    reader.close()
    return content, factor


def _contours_for_boxes(boxes):
    contours = []
    for box in boxes:
        contours.append(np.array((
                                (box[0],box[1]),
                                (box[0]+box[2],box[1]),
                                (box[0]+box[2],box[1]+box[3]),
                                (box[0],box[1]+box[3])
                                )))
    return contours

def _create_wsi_reader(wsi_path, spacing, cache_dir='./cache', spacing_tolerance=0.25):
    """ creates the reader, converts the wsi, if spacing is missing """
    # todo cache
    reader = create_reader(wsi_path, spacing_tolerance=spacing_tolerance)
    wsi_out_spacing = reader.refine(spacing)
    return reader


def _create_slide_arr(packed_height, packed_width, cache_dir, name='slide_arr.dat',
                      dims=3, fill=255, dtype=np.uint8, mem_slide=False):
    if mem_slide:
        slide_arr = np.ones((packed_height, packed_width, 3), dtype=np.uint8) * fill
    else:
        mkdir(cache_dir)
        cache_path = str(Path(cache_dir)/name)
        print('creating slide_array... %s' % cache_path)
        if dims==1:
            shape =(packed_height, packed_width)
        else:
            shape =(packed_height, packed_width, dims)
        slide_arr = np.memmap(cache_path, dtype=dtype, mode='w+', shape=shape)
        print('filling slide array..')
        if fill is not None:
            slide_arr.fill(fill)
        print('slide_arr created')
    return slide_arr


def _upscale_masks(wsi_name_path_map, wsi_name_mask_map, mask_spacing, cache_dir, spacing_tolerance=0.3):
    """" if mask spacing is not present (too low), upscales the mask and copies them to cache_dir
     return map name->mask_spacing """
    mask_spacings_map = {}
    for name,mpath in wsi_name_mask_map.copy().items():
        mreader = create_reader(mpath, spacing_tolerance=spacing_tolerance)
        try:
            mask_spacing = mreader.refine(mask_spacing)
        except InvalidSpacingError as ex:
            if cache_dir is None: raise ValueError('upscale masks requires cache_dir')
            mkdir(cache_dir)
            print('upsaling %s to spacing %f' % (str(mpath), mask_spacing))
            cdir = Path(cache_dir)/'tissue_masks'
            ensure_dir_exists(cdir)
            cpath = cdir/Path(mpath).name
            mask_spacing = resize_mask(mpath, wsi_name_path_map[name], out_path=cpath, spacing=mask_spacing, spacing_tolerance=spacing_tolerance)
            wsi_name_mask_map[name] = str(cpath)
        mask_spacings_map[name] = mask_spacing
    return mask_spacings_map

def _get_tissue_masks_out_dir(out_dir, masks_out_dir=None, tissue_masks_dir_name='tissue_masks'):
    if masks_out_dir is None:
        masks_out_dir = str(Path(out_dir).parent/tissue_masks_dir_name)
    return masks_out_dir



def pack_slide(wsi_pathes, mask_pathes, out_dir, spacing=None, level=0, out_name=None, cache_dir=None,
               processing_spacing=4, mask_spacing=4, min_area=0.01, tile_size=512, overwrite=False, orig_anno_dir=None,
               packed_anno_dir=None, tissue_masks_dir_name='tissue_masks', packed_dir_name='images',
               box_margin=1, thumbnails=True, mem_slide=False,
               spacing_tolerance=0.3, clear_locks=False, mask_label=None, nolinks=False,
               # out_format='tif', writer='asap'
               ):
    # if writer not in ['asap','pyvips']:
    #     raise ValueError('unkonwn writer %s' % writer)
    # is_tif = 'tif' in out_format
    if spacing is None: #take the spacing from the first slide
        reader = create_reader(wsi_pathes[0])
        spacing = reader.spacings[level]
        reader.close()
    elif spacing not in allowed_spacings or processing_spacing not in allowed_spacings or mask_spacing not in allowed_spacings:
        print('Warning: strange spacings:', spacing, mask_spacing, processing_spacing)

    #1. get background mask - polygon-boxes
    #2. pack them with rpack
    #3. write result with asap
    #4. create xml-annos for the rectangles for the original and packed slide
    #group is the slide name, anno name is top_left, bottom_right

    masks_out_dir = Path(out_dir)/tissue_masks_dir_name

    if orig_anno_dir is None:
        # orig_anno_dir = str(out_dir)+'_orig_anno'
        orig_anno_dir = str(Path(out_dir)/'orig_tissue_anno')
    if packed_anno_dir is None:
        packed_anno_dir = str(Path(out_dir)/'tissue_anno')
    mkdirs(out_dir, orig_anno_dir, packed_anno_dir)
    orig_anno_links_dir = orig_anno_dir+'_links'
    packed_anno_links_dir = packed_anno_dir+'_links'

    mask_pathes = [str(mp) for mp in mask_pathes]
    wsi_name_mask_map = {Path(wp).stem:mp for wp,mp in zip(wsi_pathes,mask_pathes)}
    wsi_name_path_map = {Path(wp).stem:wp for wp in wsi_pathes}
    wsi_names = list(sorted(wsi_name_path_map.keys()))
    print('packing %d slides:' % len(wsi_names), sorted(list(sorted(wsi_name_path_map.values()))))
    print('masks:', sorted(list(wsi_name_mask_map.items())))
    if out_name is None:
        if len(wsi_names)>1:
            out_name = '__'.join(wsi_names)
        else:
            out_name = wsi_names[0]

    out_path = Path(out_dir)/packed_dir_name/out_name
    mkdir(out_path.parent)
    if not str(out_path).endswith('tif'):
        out_path = Path(str(out_path)+'.tif')

    if out_path.exists():
        if overwrite:
            print('overwriting %s' % out_path)
            Path(out_path).unlink()
        else:
            print('skipping existing %s' % str(out_path))
            return

    lock_path = Path(str(out_path)+'.lock')
    if lock_path.exists() and not clear_locks:
        print('skipping locked %s' % str(lock_path))
        return
    else:
        lock_path.touch()
        path_unlinker = ExitHandler.instance().add_path_unlinker(lock_path)

    if cache_dir is None:
        cache_dir = Path(out_dir)/'cache'
    mask_spacing_map = _upscale_masks(wsi_name_path_map, wsi_name_mask_map, mask_spacing, cache_dir, spacing_tolerance=spacing_tolerance)

    # out_to_mask_factor = spacing/mask_spacing
    out_to_mask_factor = spacing/max(mask_spacing_map.values())
    processing_to_mask_factor = processing_spacing/mask_spacing
    try:
        skipped_bad_sp = []
        ensure_dir_exists(out_dir)
        readers = []
        wsi_boxindices_map = {} #wsi_path -> [box-ind in all_boxes list]
        wsi_out_spacings = []
        #boxes and for each box the corresponding wsi-reader and wsi-path
        all_boxes = [];  all_boxes_wsi_pathes = [];  all_out_factors = []; all_box_names = []
        all_boxes_readers = []; all_mask_pathes = []
        for wsi_name in wsi_names:
            wsi_path = wsi_name_path_map[wsi_name]
            print('packing %s' % wsi_path)
            reader = _create_wsi_reader(wsi_path, spacing, cache_dir=cache_dir, spacing_tolerance=spacing_tolerance)
            readers.append(reader)
            wsi_out_spacing = reader.refine(spacing)

            wsi_out_spacings.append(wsi_out_spacing)
            processing_spacing = reader.refine(processing_spacing)
            _, wsi_factor = cmp_shape_spacing_factor(reader, processing_spacing)
            wsi_out_factor = processing_spacing / wsi_out_spacing
            wsi_out_shape = reader.shapes[reader.level(processing_spacing)]
            # wsi/reader: (height, width) or (rows, cols) -> array-format
            print('%s: out_spacing=%.2f shape=%s, processing at %.2f shape=%s, factor_wsi=%.1f, factor out=%.1f' % \
                  (wsi_name, wsi_out_spacing, str(reader.shapes[reader.level(wsi_out_spacing)]),
                   processing_spacing, str(wsi_out_shape), wsi_factor, wsi_out_factor))

            mp = wsi_name_mask_map[wsi_name]
            boxes, large_areas = _create_tissue_boxes(mp, spacing=processing_spacing, min_area=min_area, box_margin=box_margin,
                                                      mask_label=mask_label)
            #boxes: [(x,y,width,height)]  #image-format
            if len(boxes)==0:
                print("skipping %s, didn't find tissue sections>%.2f" % (wsi_name, min_area) )
            else:
                print('found %d boxes, min area=%.3f, max area=%.3f' % (len(boxes), np.min(large_areas), np.max(large_areas)))
            #create asap annotations for the original slides
            box_names = _create_box_names(boxes, large_areas, wsi_factor)
            all_box_names.extend(box_names)
            box_contours = _contours_for_boxes(boxes)
            orig_anno_path = Path(orig_anno_dir)/(Path(wsi_path).stem+'.xml')
            #overwrite in case previous failed
            write_anno_for_contours(box_contours, names=box_names, wsi_path=reader, anno_path=orig_anno_path,
                                    contour_spacing=processing_spacing, add_area_to_contour_name=False, overwrite=True)
            if not nolinks:
                make_asap_link(wsi_path, mask_path=None, anno_path=orig_anno_path, links_dir=orig_anno_links_dir)

            # keep everything in the order of the boxes
            all_boxes.extend(boxes)
            for b in boxes:
                all_boxes_readers.append(reader)
                all_boxes_wsi_pathes.append(wsi_path)
                all_out_factors.append(wsi_out_factor)
                all_mask_pathes.append(mp)


        # pack boxes
        box_sizes, positions = _pack_boxes(all_boxes)
        packed_boxes = []
        for i,pos in enumerate(positions):
            packed_boxes.append([pos[0],pos[1],box_sizes[i][0],box_sizes[i][1]])

        # convert to out spacing
        packed_boxes_out_sp = [np.array(pbox)*all_out_factors[i] for i,pbox in enumerate(packed_boxes)] #at out spacing
        packed_boxes_out_sp = np.rint(packed_boxes_out_sp).astype(np.uint)
        all_boxes_out_sp = np.rint([box*all_out_factors[i] for i,box in enumerate(all_boxes)]).astype(np.uint)

        #create packed slide
        packed_height, packed_width = _determine_slide_size(packed_boxes_out_sp, tile_size) # determine out slide shape

        #wsi-reader coordinates are in array-format, opencv is in image-format
        cache_wsi_dir = Path(cache_dir)/out_name
        slide_arr = _create_slide_arr(packed_height, packed_width, cache_wsi_dir, mem_slide=mem_slide)
        print('process %d boxes...' % len(packed_boxes_out_sp))
        runtimer = RunEst(n_tasks=len(packed_boxes_out_sp), print_fct=print)
        last_time = time.time()
        for i, pbox in enumerate(packed_boxes_out_sp):
            start_time = time.time()
            if start_time - last_time >= 300: #log at least every 5m
                print('%d/%d boxes processed' % (i, len(packed_boxes_out_sp)))
            last_time = start_time

            runtimer.start()
            reader = all_boxes_readers[i]
            rshapex, rshapey = reader.shapes[reader.level(spacing)]
            orig_box = all_boxes_out_sp[i] #col(x), row(y), width, height
            if orig_box[2] + orig_box[3] < 1536:
                slide_arr[pbox[1]:pbox[1]+pbox[3],pbox[0]:pbox[0]+pbox[2]] =\
                    reader.read(spacing=spacing, y=int(orig_box[1]),x=int(orig_box[0]),
                                height=int(orig_box[3]),width=int(orig_box[2]))
            else:
                counter = 0
                for xd in range(int(np.ceil(orig_box[2]/tile_size))):
                    for yd in range(int(np.ceil(orig_box[3]/tile_size))):
                        # print('count: %d' % counter)
                        counter+=1
                        # print('counter', counter)
                        src_row = int(orig_box[1]+yd*tile_size)
                        src_col = int(orig_box[0]+xd*tile_size)

                        if src_row >= rshapey or src_col >= rshapex:
                            #due to the box margin it can be that row/col can go beyond slide borders
                            #which can cause an overflow error in asap when it tries to start reading
                            #from outside of the slide
                            continue

                        tile = reader.read(spacing=spacing, y=src_row, x=src_col,
                                        height=min(tile_size, int(orig_box[3])-yd*tile_size),
                                        width=min(tile_size, int(orig_box[2])-xd*tile_size))
                        tar_row = int(pbox[1]+yd*tile_size)
                        tar_col = int(pbox[0]+xd*tile_size)
                        # print('src: (%d, %d), tar: (%d, %d), tile: (%d, %d), dtype: %s' % \
                        #       (src_row, src_col, tar_row, tar_col, tile.shape[0], tile.shape[1], str(tile.dtype)))
                        slide_arr[tar_row:tar_row+tile.shape[0], tar_col:tar_col+tile.shape[1]] = tile

            runtimer.stop(print_remaining_string=i>0 and i%5==0 and i<len(packed_boxes_out_sp)-1)

        out_spacing = np.mean(wsi_out_spacings)
        print('writing packed slide of shape %s at spacing %.1f' %\
              (str(slide_arr.shape), out_spacing))
        cache_out_path = Path(cache_wsi_dir)/'out'/Path(out_path).name
        ensure_dir_exists(cache_out_path.parent)
        with timer('writing...'):
            writer = ArrayImageWriter()
            writer.write_array(slide_arr, path=cache_out_path, spacing=out_spacing)
        print(cache_out_path)
        print('del slide_arr')
        del slide_arr
        print('closing readers')
        for reader in readers:
            reader.close()

        print('create annotations for the packed slide with %d boxes (group per slide)' % len(packed_boxes_out_sp))
        packed_box_contours = _contours_for_boxes(packed_boxes_out_sp)
        packed_contours_map = defaultdict(list)
        packed_names_map = defaultdict(list)
        for i,pbc in enumerate(packed_box_contours):
            wsi_name = Path(all_boxes_wsi_pathes[i]).stem
            packed_contours_map[wsi_name].append(pbc)
            packed_names_map[wsi_name].append(all_box_names[i])
        packed_anno_path = Path(packed_anno_dir)/(out_name+'.xml')
        print("writing anno %s" % str(packed_anno_path))
        write_anno_for_contours(packed_contours_map, names=packed_names_map, #wsi_path=cache_out_path,
                                anno_path=packed_anno_path, wsi_spacing=out_spacing,
                                contour_spacing=out_spacing, add_area_to_contour_name=False, overwrite=True)
        # make_asap_link(wsi_path, mask_path=None, anno_path=orig_anno_path, links_dir=orig_anno_links_dir)

        if thumbnails:
            thumb_dir = Path(out_dir)/'thumbnails'
            ensure_dir_exists(thumb_dir)
            thumb_path = Path(thumb_dir)/(Path(cache_out_path).stem+'.jpg')
            create_thumbnail(cache_out_path, thumb_path, overwrite=True, openslide=False)

        print('copying packed slide %s to %s' % (str(cache_out_path), out_dir))
        shutil.copyfile(str(cache_out_path), str(out_path))

        print('write masks to %s' % str(masks_out_dir))
        ensure_dir_exists(masks_out_dir)
        mask_out_path = Path(masks_out_dir)/(out_name+'_tissue.tif')
        # mask_height = int(round(packed_height/np.mean(all_out_factors)))
        # mask_width = int(round(packed_width/np.mean(all_out_factors)))

        mask_height = int(round(packed_height*out_to_mask_factor))
        mask_width = int(round(packed_width*out_to_mask_factor))
        print('packed size: (%d, %d), mask size: (%d, %d)' % (packed_height, packed_width, mask_height, mask_width))

        mspacings = []
        mask_path_reader_map = {}
        for mp in all_mask_pathes:
            if mp not in mask_path_reader_map:
                mask_path_reader_map[mp] = create_reader(mp, spacing_tolerance=spacing_tolerance)
                mspacings.append(mask_path_reader_map[mp].refine(mask_spacing)) #should be very similar
        # mspacing = np.mean(mspacings) #if they are different, will probably result in out of bound errors for the mask
        # mask_out_factor = mspacing/processing_spacing

        # mask_arr = np.zeros((mask_height, mask_width, 1), dtype=np.uint8)
        mask_arr = _create_slide_arr(mask_height, mask_width, cache_dir=cache_wsi_dir, name='mask_arr.dat',
                                     dims=1, fill=0, mem_slide=mem_slide)
        print('process %d packed boxes' % len(packed_boxes))
        for i, pbox in enumerate(packed_boxes):
            mreader = mask_path_reader_map[all_mask_pathes[i]]
            # mspacing = mreader.refine(mask_spacing)
            #pbox and orig_box are at processing spacing
            orig_box_mask_sp = np.round(np.array(all_boxes[i])*processing_to_mask_factor).astype(np.uint32)
            pbox = np.round(np.array(pbox)*processing_to_mask_factor).astype(np.uint32)
            # mask_arr[pbox[1]:pbox[1] + pbox[3], pbox[0]:pbox[0] + pbox[2]] = \
            #     mreader.read(processing_spacing, int(orig_box[1]), int(orig_box[0]), int(orig_box[3]), int(orig_box[2]))
            row_up = pbox[1] + pbox[3]
            col_up = pbox[0] + pbox[2]
            if row_up > mask_arr.shape[0] or col_up > mask_arr.shape[1]:
                raise ValueError('mask ind out of bound: (%d, %d), mask:' % (row_up, col_up), mask_arr.shape[:2])
            tile = mreader.read(spacing=mask_spacing, y=int(orig_box_mask_sp[1]), x=int(orig_box_mask_sp[0]),
                                height=int(orig_box_mask_sp[3]), width=int(orig_box_mask_sp[2])).astype(np.uint8)
            if tile.shape[0]!= (row_up - pbox[1]) or tile.shape[1] != (col_up - pbox[0]):
                raise ValueError('wrong tile size:', tile.shape, 'pbox:', pbox, 'origbox:', orig_box)
            mask_arr[pbox[1]:row_up, pbox[0]:col_up] = tile

        if mask_arr.sum()==0:
            raise ValueError('empty mask!')
        print('writing mask array as tif...')
        mask_tile_size = tile_size
        if max(mask_arr.shape)<3000:
            mask_tile_size = mask_tile_size//2
        writer = ArrayImageWriter(tile_size=mask_tile_size)
        # writer.write_array(mask_arr, path=mask_out_path, spacing=processing_spacing)
        mspacing = np.mean(mspacings)
        writer.write_array(mask_arr, path=mask_out_path, spacing=mspacing)
        print('closing mreaders...')
        for mreader in mask_path_reader_map.values():
            mreader.close()
        if str(out_path).endswith('tif') and not nolinks:
            make_asap_link(out_path, mask_path=mask_out_path, anno_path=packed_anno_path, links_dir=packed_anno_links_dir)

        print('deleting cache dir %s' % cache_wsi_dir)
        shutil.rmtree(str(cache_wsi_dir))
        print('Done %s' % out_name)
    except:
        print(traceback.print_exc())
        print('failed packing wsi: %s, masks: %s' % (str(wsi_pathes), str(mask_pathes)))
        raise
    finally:
        lock_path.unlink()
        ExitHandler.instance().remove(path_unlinker)
        print('lock_path %s unlinked' % str(lock_path))




def _pack_boxes(all_boxes, size_diff_factor = 0.5):
    box_sizes = [(int(box[2]), int(box[3])) for box in all_boxes]
    # box_sizes = [box[2:] for box in all_boxes]
    positions = rpack.pack(box_sizes)  # at processing spacing
    packed_size = rpack.bbox_size(box_sizes, positions)  # width, height
    print('rpack packed size:', packed_size)

    print('trying to enforce rectangle size...')
    counter=0
    for sdf in np.arange(size_diff_factor, 0.96, 0.05):
        # asymmetry_factor = min(packed_size)/max(packed_size)
        # if asymmetry_factor < sdf:
        rparams = {'max_height': int(max(packed_size) * sdf),
                    'max_width': int(max(packed_size) * sdf)}
        try:
            # print('trying to enforce rpack reducing size by %.2f %s' % (sdf, str(rparams)))
            positions = rpack.pack(box_sizes, **rparams)  # at processing spacing
            packed_size = rpack.bbox_size(box_sizes, positions)
            print('corrected packed size:', packed_size)
            break
        except rpack.PackingImpossibleError as ex:
            counter+=1
    print('final size after %d trials' % counter, packed_size)
    return box_sizes, positions


def _determine_slide_size(packed_boxes, tile_size):
    width = 0; height = 0
    for i, pbox in enumerate(packed_boxes):
        width_i = pbox[0] + pbox[2]
        height_i = pbox[1] + pbox[3]
        if width_i > width:
            width = width_i
        if height_i > height:
            height = height_i
    width = int(np.ceil(width / tile_size) * tile_size)
    height = int(np.ceil(height / tile_size) * tile_size)
    return height, width


def _create_box_names(boxes, areas, wsi_factor):
    box_names = []
    for b, box in enumerate(boxes):
        box = np.array(box) * wsi_factor
        box = np.rint(box).astype(np.int64)
        box_names.append(f'{box[0]}_{box[1]}_{box[2]}_{box[3]}_area_{areas[b]:.1f}')
    return box_names


def _create_tissue_boxes(mask_path, spacing, min_area=0, box_margin=0, mask_label=None, verbose=False):
    print('reading content from %s at spacing %.2f' % (str(mask_path), spacing))
    content, _ = _read_content(mask_path, spacing)
    print('finding contours for shape %s' % str(content.shape))
    if mask_label is None:
        contours = find_contours(content > 0)
    else:
        contours = find_contours(content == mask_label)
    if len(contours)==0:
        raise ValueError('no tissue in mask %s' % str(mask_path))

    areas = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        area_mm = px_to_mm2(area, spacing)
        areas.append(area_mm)

    if np.max(areas)<min_area:
        old_min_area = min_area
        min_area = np.median(areas)
        print('all sections in %s < %.2f, changing min_area to %.2f' % (Path(mask_path).stem, old_min_area, min_area))

    large_contours = []; large_areas = []
    for i,cnt in enumerate(contours):
        area_mm = areas[i]
        if area_mm >= min_area:
            large_contours.append(cnt)
            large_areas.append(area_mm)

    if len(large_areas)==0: #shouldnt happen unless completely zero section, but just in case
        return [], []

    boxes = [cv.boundingRect(c) for c in large_contours]
    merge_overlapping_bboxes(boxes)
    boxes = np.array(boxes)
    boxes[:, 2] += box_margin
    boxes[:, 3] += box_margin
    return boxes, large_areas



def _pack_slide_wrapper(wsi_pathes, **kwargs):
    try:
        pack_slide(wsi_pathes=wsi_pathes, **kwargs)
        return 'ok'
    except Exception as ex:
        return str(wsi_pathes)+str(ex)
    except:
        return str(wsi_pathes)+str(sys.exc_info())

def pack_slides(data, out_dir, packed_name_col='name', packed_dir_name='images', path_col='path', mask_col='mask_path',
                mask_dir=None, random_order=False, cpus=0, overwrite=False, tissue_masks_dir_name='tissue_masks',
                check_previous_params=False, cache_dir=None,
                anno_dir=None, anno_out_dir=None, anno_out_dir_prefix=None, **kwargs):
    """ slide_pack_mapping: either csv-path or df with cols for packed_name, wsi path and optionally mask path """
    out_dir = Path(out_dir)
    if cache_dir is None:
        cache_dir = out_dir/'cache'
    out_dir_packed = out_dir/packed_dir_name
    mkdir(out_dir_packed)
    param_info = ParamInfo(out_dir, filename='pack_args.yaml', overwrite=not check_previous_params)
    param_info.save(**kwargs)

    if isinstance(data, pd.DataFrame):
        df = data
    elif data.endswith('csv'):
        df = pd.read_csv(data)
    elif data.endswith('xlsx'):
        df = pd.read_excel(data)
    else:
        wsi_pathes = PathUtils.list_pathes(data, ending=['svs','tif', 'tiff', 'mrxs', 'ndpi', 'dicom'], ret='str', sort=True)
        names = [Path(wp).stem for wp in wsi_pathes]
        df = pd.DataFrame({packed_name_col:names, path_col:wsi_pathes})
    if packed_name_col not in df:
        pot_names = [Path(p).stem for p in df[path_col].values]
        if len(set(pot_names))==len(pot_names):
            print('using image name for packed results')
            df[packed_name_col] = pot_names

    if path_col not in df:
        raise  ValueError('missing path column %s' % path_col)
    if packed_name_col not in df:
        raise  ValueError('missing packed_name_col column %s' % packed_name_col)

    df[packed_name_col] = df[packed_name_col].astype(str)
    all_wsi_pathes = list(df[path_col])

    if mask_col not in df:
        if mask_dir is None: #TODO: just throw an error
            mask_dirs = []
            wsi_dirs = [str(Path(p).parent) for p in df[path_col]]
            wsi_dirs = list(set(wsi_dirs))
            for wdir in wsi_dirs.copy():
                pot_mask_dirs = [wdir+'_'+tissue_masks_dir_name, Path(wdir).parent/tissue_masks_dir_name,
                                 Path(wdir)/tissue_masks_dir_name]
                found = False
                for pot_mask_dir in pot_mask_dirs:
                    if Path(pot_mask_dir).exists():
                        found = True
                        break
                if not found:
                    print('skipping wsi_dir due to missing tissue masks')
                    wsi_dirs.remove(wdir)
                    continue
                mask_dirs.append(pot_mask_dir)
            if len(wsi_dirs)==0 or len(mask_dirs)==0:
                raise ValueError('no wsi dir or mask dir!')

            mask_pathes = []
            mask_dirs = list(set(mask_dirs))
            print('auto determined %d mask_dirs: %s' % (len(mask_dirs), str(mask_dirs)))
            for mdir in mask_dirs:
                found_masks = PathUtils.list_pathes(mdir, ending='.tif', type='file', ret='str')
                print('found %d masks in %s' % (len(found_masks), str(mdir)))
                mask_pathes.extend(found_masks)
            if len(mask_dirs)>1:
                print('found overall %d masks' % len(mask_pathes))
        else:
            mask_pathes = PathUtils.list_pathes(mask_dir, ending='.tif', ret='str')
            print('found %d masks in %s' % (len(mask_pathes), str(mask_dir)))
        if mask_pathes is None or None in mask_pathes:
            raise ValueError('error determining mask_pathes from mask_dir %s' % str(mask_dir))
        wpathes = df[path_col]
        _, mask_pathes = get_corresponding_pathes(wpathes, mask_pathes, must_all_match=False,
                                               take_shortest=True, as_string=True)
        mask_col = 'mask_path'
        df[mask_col] = mask_pathes

    skipped = []
    all_pack_params = []
    n_packed = df[packed_name_col].nunique()
    for packed_name, group in df.groupby(packed_name_col):
        wsi_pathes_group = group[path_col]
        mask_pathes_group = group[mask_col]
        # wsi_pathes_group, mask_pathes_group = get_matching_pathes(wsi_pathes_group, mask_pathes)
        if mask_pathes_group is None or None in mask_pathes_group or None in list(mask_pathes_group):
            print('no matching mask for %s' % str(wsi_pathes_group))
            print(wsi_pathes_group)
            print(mask_pathes_group)
            skipped.append(packed_name)
            continue
        # print('pack_slide kwargs', kwargs)
        pack_params = dict(wsi_pathes=list(wsi_pathes_group), mask_pathes=list(mask_pathes_group),
                    packed_dir_name=packed_dir_name,
                    tissue_masks_dir_name=tissue_masks_dir_name, cache_dir=cache_dir,
                    out_dir=out_dir, out_name=str(packed_name), overwrite=overwrite, **kwargs)
        all_pack_params.append(pack_params)

    if random_order:
        print('processing in random order')
        random.shuffle(all_pack_params)

    print('packing %d slides to %d' % (len(df), n_packed))

    results = multiproc_pool2(_pack_slide_wrapper, all_pack_params, cpus=cpus)

    failures = [r for r in results if r!='ok']
    if len(skipped)>0:
        print('%d skipped: %s' % (len(skipped), str(skipped)))
    if len(failures)!=0:
        print('%d failures:' % len(failures))
        print(failures)

    if len(all_pack_params)>0 and len(failures)==0:
        if check_packed(out_dir_packed):
            create_slides_info_summary(out_dir_packed, out_dir=out_dir/'slide_summary', overwrite=True)
            create_slides_info_summary(out_dir/tissue_masks_dir_name, overwrite=True)

    if anno_dir is not None:
        pack_annos(anno_dir, out_dir_packed, all_wsi_pathes,
                   out_dir=anno_out_dir, out_dir_prefix=anno_out_dir_prefix, overwrite=overwrite)

    create_pathes_csv(out_dir_packed, out_path=out_dir/'pathes.csv')
    if cache_dir is not None:
        try:  os.rmdir(str(cache_dir));
        except: pass
    print('Done!')


def check_packed(packed_image_dir):
    """ sometimes the slide is packed, but no tissue masks are created (crashed process?)
     For now, delete all those slides (to re-pack)
     packed_dir: directory with the packed results (seven dirs or so)
     return 1 if ok
     """
    packed_image_dir = Path(packed_image_dir)
    tissue_dir = packed_image_dir.parent/'tissue_masks'
    print('checking for missing tissue masks in %s' % str(tissue_dir))
    if not tissue_dir.exists(): raise ValueError('no tissue dir %s' % str(tissue_dir))
    if not packed_image_dir.exists(): raise ValueError('no image dir %s' % str(packed_image_dir))

    # all_dirs = PathUtils.list_pathes(packed_dir, type='dir')
    packed_slides = PathUtils.list_pathes(packed_image_dir, containing_or=['.tif','.jpg','.jpeg'])
    masks = PathUtils.list_pathes(tissue_dir, ending='.tif')
    found_slides, found_masks = get_matching_pathes(packed_slides, masks, ignore_missing=True, same_name=True,
                                                    replacements={'_tissue':''})
    failed = list_not_in(packed_slides, found_slides)

    for f in failed:
        print('rm %s' % str(f))

    print('%d slides, %d failed' % (len(packed_slides), len(failed)))
    return len(failed)==0



def main():
    print('sys.path:', sys.path)
    parser = FlexArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='wsi-dir or csv with three columns, one'
                                            'for the name of the packed slide (default:"name") and the wsi path (default:"path"')
    parser.add_argument('--packed_name_col', type=str, required=False, default='name')
    parser.add_argument('--path_col', type=str, default='path')
    parser.add_argument('--mask_col', type=str, default='mask_path')
    parser.add_argument('--mask_dir', type=str, default=None, help='alternative to specifying masks in mask_col,'
                                        'if None tries to find masks automatically looking for "tissue_masks"')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--spacing', type=float, required=False)
    parser.add_argument('--level', type=int, required=False)
    parser.add_argument('--processing_spacing', type=float, required=False, default=4, help='spacing at which the tissue is detected, 4 or 8 recommended')
    parser.add_argument('--mask_spacing', type=float, required=False, default=4)
    parser.add_argument('--min_area', type=float, required=False, default=0.4, help='min section tissue area in mm2')
    parser.add_argument('--cache_dir', type=str, required=False, help='defaults to out_dir/cache')
    parser.add_argument('--anno_dir', type=str, required=False, default=None, help='annotations belonging to the slides to be packed')
    parser.add_argument('--sleep', type=int, default=0, required=False, help='for debugging, letting the docker sleep before doing anything')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing results')
    parser.add_argument('--cpus', type=int, required=False, default=0, help='for multiproc')
    parser.add_argument('--random_order', action='store_true', help='random order of processing')
    # parser.add_argument('--writer', type=str, required=False, default='asap', choices=['asap','pyvips'])

    args = parser.parse_args()
    if not is_dict(args):
        args = vars(args)

    sleep = args.pop('sleep',0)
    if sleep > 0: print('sleeping %ds' % sleep)
    time.sleep(sleep)

    print('args:', args)
    pack_slides(**args)


if __name__ == '__main__':
    main()