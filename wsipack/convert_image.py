import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from wsipack.utils.cool_utils import mkdir, take_closest_number_index
from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.asap_writer import AsapMaskWriter, AsapImageWriter, ArrayImageWriter
from wsipack.wsi.wsi_read import create_reader


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
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]  # drop alpha, AsapImageWriter expects RGB
    writer = ArrayImageWriter(tile_size=tile_size)
    writer.write_array(arr=arr, path=path, spacing=spacing)

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

    heights = [hw[1] for hw in shapes]
    level = take_closest_number_index(heights, h)
    print('for mask shape(%d,%d) closest level %d with spacing  %.3f, shape %s' % \
          (h, w, level, spacings[level], str(shapes[level])))

    spacing = spacings[level]
    level_shape = shapes[level] #(width, height)
    if (w, h) != level_shape:
        print('resizing mask from (%d,%d) to level size %s' % (w, h, str(level_shape)))
        if mask_arr.dtype == bool: #mode '1' pngs, cv2 can't resize bool arrays
            mask_arr = mask_arr.astype(np.uint8)
        mask_arr = cv2.resize(mask_arr, level_shape, interpolation=cv2.INTER_NEAREST)
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


def _process_args_tif(data, out_dir, path_col, spacing, overwrite):
    print('convert images to tif')
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


def _process_args_mask(data, out_dir, path_col, slide_path, slide_col, slide_dir, overwrite):
    print('convert masks to tif')
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
        _process_args_tif(**args)
    elif mode == 'mask':
        _process_args_mask(**args)


if __name__ == '__main__':
    main()

