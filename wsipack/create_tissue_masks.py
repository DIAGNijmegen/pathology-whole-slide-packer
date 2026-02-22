#adapted from: Pingjun Chen
# https://github.com/PingjunChen/tissueloc/blob/master/tissueloc/locate_tissue.py
# MIT License Copyright (c) 2018 Pingjun Chen

## Changes: output asap tif masks, support for spacing parameter
## Note: fat and faint stroma are may not be detected

import argparse
import sys
from pathlib import Path

from scipy.ndimage import binary_fill_holes
from skimage import filters

from wsipack.wsi.asap_writer import ArrayImageWriter
from wsipack.wsi.wsi_read import ImageReader, create_reader

try:
    from skimage import img_as_ubyte
except:
    from skimage.util import img_as_ubyte
from skimage.morphology import remove_small_objects
import cv2

import openslide
from PIL import Image
import numpy as np
from tqdm import tqdm

from wsipack.utils.cool_utils import mkdir
from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.contour_utils import mm2_to_px


def select_slide_level(slide_path, max_size=2048):
    """Find the slide level to perform tissue localization
    Parameters
    ----------
    slide_path : valid slide path
        The slide to process.
    max_size : int
        Max height and width for the size of slide with selected level
    Returns
    -------
    level : int
        Selected level.
    d_factor: int
        Downsampling factor of selected level compared to level 0
    Notes
    -----
    The slide should have hierarchical storage structure.
    Examples
    --------
    """

    slide_head = openslide.open_slide(slide_path)
    level_dims = slide_head.level_dimensions

    d_factors = slide_head.level_downsamples
    # assert len(level_dims) > 1, "This slide doesnot have mutliple levels"
    select_level = len(level_dims) - 1
    for ind in np.arange(len(level_dims)):
        cur_w, cur_h = level_dims[ind]
        if cur_w < max_size and cur_h < max_size:
            select_level = ind
            break

    d_factor = int(d_factors[select_level])

    return select_level, d_factor



def load_slide_img(slide_path, level=0):
    """Load slide image with specific level
    Parameters
    ----------
    slide_path : valid slide path
        The slide to load.
    level : int
        Slide level to load.
    Returns
    -------
    slide_img : np.array
        Numpy matrix with RGB three channels.
    Notes
    -----
    Whole slide image can have more than 100,000 pixels in width or height,
    small level can be very big image.
    Examples
    --------
    """

    slide_head = openslide.open_slide(slide_path)
    img_size = slide_head.level_dimensions[level]
    slide_img = slide_head.read_region((0, 0), level, img_size)
    if isinstance(slide_img, Image.Image):
        slide_img = np.asarray(slide_img)
    if slide_img.shape[2] == 4:
        slide_img = slide_img[:, :, :-1]
    return slide_img


def rgb2gray(img):
    """Convert RGB image to gray space.
    Parameters
    ----------
    img : np.array
        RGB image with 3 channels.
    Returns
    -------
    gray: np.array
        Gray image
    """
    gray = np.dot(img, [0.299, 0.587, 0.114])

    return gray


def thresh_slide(gray, thresh_val, sigma=13):
    """ Threshold gray image to binary image
    Parameters
    ----------
    gray : np.array
        2D gray image.
    thresh_val: float
        Thresholding value.
    smooth_sigma: int
        Gaussian smoothing sigma.
    Returns
    -------
    bw_img: np.array
        Binary image
    """

    # Smooth
    smooth = filters.gaussian(gray, sigma=sigma)
    smooth /= np.amax(smooth)
    # Threshold
    bw_img = smooth < thresh_val

    return bw_img


def fill_tissue_holes(bw_img):
    """ Filling holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    bw_fill: np.array
        Binary image with no holes
    """

    # Fill holes
    bw_fill = binary_fill_holes(bw_img)

    return bw_fill


def remove_small_tissue(bw_img, min_size=10000):
    """ Remove small holes in tissue image
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    min_size: int
        Minimum tissue area.
    Returns
    -------
    bw_remove: np.array
        Binary image with small tissue regions removed
    """

    bw_remove = remove_small_objects(bw_img, min_size=min_size, connectivity=8)

    return bw_remove


def find_tissue_cnts(bw_img):
    """ Fint contours of tissues
    Parameters
    ----------
    bw_img : np.array
        2D binary image.
    Returns
    -------
    cnts: list
        List of all contours coordinates of tissues.
    """

    cnts, _ = cv2.findContours(img_as_ubyte(bw_img), mode=cv2.RETR_EXTERNAL,
                               method=cv2.CHAIN_APPROX_NONE)

    return cnts


def locate_tissue(slide_path,
                       max_img_size=2048, s_level=None,
                       smooth_sigma=13,
                       thresh_val = 0.85,
                       min_tissue_size=10000):
    """ Locate tissue contours of whole slide image
    Parameters
    ----------
    slide_path : valid slide path
        The slide to locate the tissue.
    max_img_size: int
        Max height and width for the size of slide with selected level.
    smooth_sigma: int
        Gaussian smoothing sigma.
    thresh_val: float
        Thresholding value.
    min_tissue_size: int
        Minimum tissue area.
    Returns
    -------
    cnts: list
        List of all contours coordinates of tissues.
    d_factor: int
        Downsampling factor of selected level compared to level 0
    """
    # Step 1: Select the proper level
    if s_level is None:
        s_level, d_factor = select_slide_level(slide_path, max_img_size)
    else:
        d_factor = 2**s_level
    # Step 2: Load Slide image with selected level
    slide_img = load_slide_img(slide_path, s_level)
    # Step 3: Convert color image to gray
    gray_img = rgb2gray(slide_img)
    # Step 4: Smooth and Binarize
    bw_img = thresh_slide(gray_img, thresh_val, sigma=smooth_sigma)
    # Step 5: Fill tissue holes
    bw_fill = fill_tissue_holes(bw_img)
    # Step 6: Remove small tissues
    bw_remove = remove_small_tissue(bw_fill, min_tissue_size)
    # Step 7: Locate tissue regions
    # cnts = find_tissue_cnts(bw_remove)

    # slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)
    # cv2.drawContours(slide_img, cnts, -1, (0, 255, 0), 5)

    return slide_img, bw_remove, d_factor

########################### additional functions

def create_tissue_mask(slide_path, spacing=None, level=None, out_path=None, overwrite=False, min_area=0, ignore_err=False):
    if spacing is None and level is None:
        raise ValueError('specify either spacing or level')
    if out_path is not None and Path(out_path).exists() and not overwrite:
        print('not overwriting existing %s' % str(out_path))
        return

    try:
        reader = create_reader(slide_path)
        if level is None:
            level = reader.level(spacing)
        spacing = reader.spacings[level]
        reader.close()

        min_tissue_size = mm2_to_px(min_area, spacing=spacing)
        img, mask, factor = locate_tissue(str(slide_path), s_level=level, min_tissue_size=min_tissue_size)

        if out_path is not None:
            _write_tissue_mask(mask, out_path=out_path, spacing=spacing)
        return mask, spacing
    except:
        if ignore_err:
            print('failed to create mask for %s' % str(slide_path))
            print(sys.exc_info())
            return None, None
        else:
            raise

def _write_tissue_mask(mask, spacing, out_path):
    out_dir = Path(out_path).parent
    mkdir(out_dir)
    writer = ArrayImageWriter()
    writer.write_array(mask, out_path, spacing=spacing)

def create_tissue_masks(slide_dir, out_dir, spacing=4, level=None, wsi_suffix=['tif', 'tiff', 'svs', 'ndpi', 'mrxs'],
                        out_suffix='_tissue', **kwargs):
    slide_pathes = PathUtils.list_pathes(slide_dir, ending=wsi_suffix)
    print('creating %d tissue masks' % len(slide_pathes))
    if len(slide_pathes) == 0:
        print('no slides found in %s with suffix %s' % (str(slide_dir), str(wsi_suffix)))
    if out_suffix is None:
        out_suffix = ''
    for sp in tqdm(slide_pathes):
        out_path = Path(out_dir)/(sp.stem+out_suffix+'.tif')
        create_tissue_mask(sp, out_path=out_path, spacing=spacing, level=level, **kwargs)
    print('Done!')

def _process_args(args):
    print('create tissue masks with args:', args)

    wsi = args.pop('wsi')
    wsi_suffix = args.pop('wsi_suffix',None)
    if Path(wsi).is_dir():
        create_tissue_masks(slide_dir=wsi, wsi_suffix=wsi_suffix.split(','), **args)
    else:
        print('create single tissue mask for %s' % wsi)
        out_dir = args.pop('out_dir')
        out_path = Path(out_dir)/(Path(wsi).stem+args.pop('out_suffix','_tissue')+'.tif')
        create_tissue_mask(slide_path=wsi, out_path=out_path, **args)


def main():
    parser = argparse.ArgumentParser(description="Create tissue masks")
    parser.add_argument("--wsi", help="slide path or directory containing slides", required=True)
    parser.add_argument("--out_dir", help="Output directory for the tissue masks", required=True)
    parser.add_argument("--spacing", type=int, help="spacing in um/px (if not level)", required=False)
    parser.add_argument("--level", type=int, help="level (if not spacing)", required=False)
    parser.add_argument("--min_area", type=float, help="minimal tissue area in mm²", required=False, default=0.05)
    parser.add_argument("--wsi_suffix", default="tif,svs,ndpi,mrxs,dicom", help="suffix of slides in slide_dir", required=False)
    parser.add_argument("--out_suffix", default="_tissue", help="suffix for the created masks", required=False)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--ignore_err", action="store_true", help="Ignore errors")

    args = vars(parser.parse_args())

    _process_args(args)

if __name__ == '__main__':
    main()