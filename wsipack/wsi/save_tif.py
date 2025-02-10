from pathlib import Path

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.wsd_image import *


def save_tif(path, out_path, spacing=None, level=None, tile_size=512, quality=80, overwrite=False):
    if spacing is None and level is None:
        raise ValueError('spacing or level must be provided')

    if Path(out_path).exists() and not overwrite:
        print('%s already exists' % str(out_path))
        return
    reader = ImageReader(path)
    if level is not None:
        spacing = reader.spacings[0]
    else:
        spacing = reader.refine(spacing)

    shape = reader.shape(spacing)
    writer = ImageWriter(out_path, shape=(shape[0],shape[1]), spacing=spacing, quality=quality, tile_size=tile_size)
    print('writing shape %s at spacing %.2fto %s' % (str(shape), spacing, path))
    for x in tqdm(range(0, shape[0], tile_size)):
        for y in range(0, shape[1], tile_size):
            tile = reader.read(spacing=spacing, row=y, col=x, width=tile_size, height=tile_size)
            writer.write(tile, row=y, col=x)
    print('finalizing image...')
    writer.close()
    print('Done %s' % str(path), flush=True)

def _process_args(args):
    print('create tissue masks with args:', args)

    wsi = args.pop('wsi')
    wsi_suffix = args.pop('suffix',None)
    out_dir = args.pop('out_dir')
    mkdir(out_dir)
    if Path(wsi).is_dir():
        pathes = PathUtils.list_pathes(wsi, ending=wsi_suffix)
        for path in pathes:
            out_path = Path(out_dir)/(Path(wsi).stem+'.tif')
            save_tif(path, out_path=out_path, **args)
    else:
        print('convert single slide %s' % wsi)
        out_path = Path(out_dir)/(Path(wsi).stem+'.tif')
        save_tif(slide_path=wsi, out_path=out_path, **args)


def main():
    parser = argparse.ArgumentParser(description="Create tissue masks")
    parser.add_argument("--wsi", help="slide path or directory containing slides", required=True)
    parser.add_argument("--out_dir", help="Output directory for the tissue masks", required=True)
    parser.add_argument("--spacing", type=float, help="spacing in um/px (if not level)", required=False)
    parser.add_argument("--level", type=int, help="level (if not spacing)", required=False)
    parser.add_argument("--suffix", default="tif,tiff,svs,ndpi,mrxs,dicom", help="suffix of slides in slide_dir", required=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    args = vars(parser.parse_args())

    _process_args(args)

if __name__ == '__main__':
    main()