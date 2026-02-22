import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wsipack.utils.cool_utils import mkdir
from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.asap_writer import AsapMaskWriter, AsapImageWriter
from wsipack.wsi.wsi_read import create_reader


def take_closest_number(l, number):
    return min(l, key=lambda x: abs(x - number))

def take_closest_number_index(l, number):
    closest = take_closest_number(l, number)
    for ind, val in enumerate(l):
        if val==closest: return ind

def save_normal_image_as_tif(path, out_path, spacing, overwrite=False):
    """ saves a normal image (png, jpeg) as pyramidal tif """
    if Path(out_path).exists() and not overwrite:
        print('%s already exists' % str(out_path))
        return
    img = Image.open(str(path))
    arr = np.array(img).squeeze()
    img.close()
    print('converting %s -> %s (spacing=%.3f)' % (Path(path).name, out_path, spacing))
    save_array_as_image(arr, path=out_path, spacing=spacing)

def save_array_as_image(arr, path, spacing, tile_size=512):
    """ saves an array as pyramidal tif """
    if min(arr.shape[:2])<tile_size:
        tile_size = take_closest_number([8, 16, 32, 64, 128, 256], min(arr.shape[:2]))
    if len(arr.shape)==2:
        arr = arr[:,:,None]
        writer = AsapMaskWriter()
    else:
        writer = AsapImageWriter()
    shape = arr.shape
    writer.write(
        path=str(path), spacing=spacing, dimensions=(shape[1], shape[0]),
        tile_shape=(tile_size, tile_size),
    )

    for col in range(0, shape[1]+tile_size, tile_size): #+tile_size if array not divisible by tile_size
        for row in range(0, shape[0]+tile_size, tile_size):
            tile = arr[row:row+tile_size, col:col+tile_size]
            if len(tile)==0 or 0 in tile.shape: continue #for the edge-case
            if tile.shape[0]!=tile_size or tile.shape[1]!=tile_size:
                pad = ((0, tile_size-tile.shape[0]),(0, tile_size-tile.shape[1]),(0,0))
                tile = np.pad(tile, pad, mode='constant')
            writer.write_tile(tile=tile, coordinates=(col,row))  #col,row (x,y)
    writer.save()


def save_normal_image_as_mask(path, slide_path, out_path, overwrite=False):
    """ saves an image-mask (png, jpeg) as pyramidal tif for a given slide """
    if Path(out_path).exists() and not overwrite:
        print('%s already exists' % str(out_path))
        return
    reader = create_reader(str(slide_path))
    spacings = reader.spacings
    shapes = reader.shapes
    reader.close()

    mask = Image.open(str(path))
    mask_arr = np.array(mask)
    h,w = mask.height, mask.width
    mask.close()

    heights = [hw[0] for hw in shapes]
    level = take_closest_number_index(heights, h)
    print('for mask shape(%d,%d) closest level %d with spacing  %.3f, shape %s' % \
          (h, w, level, spacings[level], str(shapes[level])))

    spacing = spacings[level]
    return save_array_as_image(mask_arr, spacing=spacing, path=out_path)


IMAGE_SUFFIXES = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff']

def _resolve_inputs(data, path_col='path'):
    """ Resolves data argument to a DataFrame. Supports single path, directory, or csv/xlsx. """
    data = str(data)
    if data.endswith('csv'):
        df = pd.read_csv(data)
    elif data.endswith('xlsx'):
        df = pd.read_excel(data)
    elif Path(data).is_dir():
        pathes = PathUtils.list_pathes(data, ending=IMAGE_SUFFIXES, ret='str', sort=True)
        print('found %d images in %s' % (len(pathes), data))
        names = [Path(p).stem for p in pathes]
        df = pd.DataFrame({'name': names, path_col: pathes})
    else:
        # single file
        df = pd.DataFrame({'name': [Path(data).stem], path_col: [data]})
    return df


def _process_args_tif(args):
    print('convert images to tif with args:', args)
    data = args.pop('data')
    out_dir = args.pop('out_dir')
    path_col = args.pop('path_col')
    spacing = args.pop('spacing')
    overwrite = args.pop('overwrite')
    mkdir(out_dir)

    df = _resolve_inputs(data, path_col=path_col)
    if path_col not in df:
        raise ValueError('missing path column "%s" in data' % path_col)

    print('converting %d images' % len(df))
    for _, row in df.iterrows():
        path = row[path_col]
        out_path = str(Path(out_dir) / (Path(path).stem + '.tif'))
        save_normal_image_as_tif(path, out_path=out_path, spacing=spacing, overwrite=overwrite)
    print('Done!')


def _process_args_mask(args):
    print('convert masks to tif with args:', args)
    data = args.pop('data')
    out_dir = args.pop('out_dir')
    path_col = args.pop('path_col')
    slide_path = args.pop('slide_path')
    slide_col = args.pop('slide_col')
    slide_dir = args.pop('slide_dir')
    overwrite = args.pop('overwrite')
    mkdir(out_dir)

    df = _resolve_inputs(data, path_col=path_col)
    if path_col not in df:
        raise ValueError('missing path column "%s" in data' % path_col)

    if slide_col in df:
        # slide paths from CSV
        pass
    elif slide_path is not None:
        # single slide path for all entries
        df[slide_col] = slide_path
    elif slide_dir is not None:
        # match slides from directory by name
        slide_pathes = PathUtils.list_pathes(slide_dir, ending=['svs', 'tif', 'tiff', 'mrxs', 'ndpi', 'dicom'], ret='str', sort=True)
        slide_map = {Path(sp).stem: sp for sp in slide_pathes}
        matched = []
        for _, row in df.iterrows():
            name = Path(row[path_col]).stem
            if name in slide_map:
                matched.append(slide_map[name])
            else:
                print('WARNING: no matching slide for %s' % name)
                matched.append(None)
        df[slide_col] = matched
    else:
        raise ValueError('for mask mode, provide either slide_col in csv or --slide_dir')

    print('converting %d masks' % len(df))
    for _, row in df.iterrows():
        path = row[path_col]
        slide_path = row[slide_col]
        if slide_path is None or (isinstance(slide_path, float) and np.isnan(slide_path)):
            print('skipping %s (no slide)' % path)
            continue
        out_path = str(Path(out_dir) / (Path(path).stem + '.tif'))
        save_normal_image_as_mask(path, slide_path=slide_path, out_path=out_path, overwrite=overwrite)
    print('Done!')


def main():
    parser = argparse.ArgumentParser(description="Convert normal images (png, jpeg) to pyramidal tif")
    subparsers = parser.add_subparsers(dest='mode', help='conversion mode', required=True)

    # common arguments added to both subparsers
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", required=True, help="image path, directory, or csv/xlsx file")
    common.add_argument("--out_dir", required=True, help="output directory for the tif files")
    common.add_argument("--path_col", default="path", help="column name for image paths in csv")
    common.add_argument("--overwrite", action="store_true", help="overwrite existing files")

    # tif subcommand
    parser_tif = subparsers.add_parser('tif', parents=[common], help='convert image to pyramidal tif')
    parser_tif.add_argument("--spacing", type=float, required=True, help="spacing in um/px")

    # mask subcommand
    parser_mask = subparsers.add_parser('mask', parents=[common], help='convert mask image to pyramidal tif using slide spacing')
    parser_mask.add_argument("--slide_path", default=None, help="slide path (for single file input)")
    parser_mask.add_argument("--slide_dir", default=None, help="directory with slides to match masks against (by name)")
    parser_mask.add_argument("--slide_col", default="slide_path", help="column name for slide paths in csv")

    args = vars(parser.parse_args())
    mode = args.pop('mode')

    if mode == 'tif':
        _process_args_tif(args)
    elif mode == 'mask':
        _process_args_mask(args)


if __name__ == '__main__':
    main()

