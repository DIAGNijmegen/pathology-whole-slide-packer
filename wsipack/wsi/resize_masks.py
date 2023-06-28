import shutil, tqdm
from pathlib import Path

import cv2

from wsipack.utils.cool_utils import ensure_dir_exists
from wsipack.utils.path_utils import PathUtils, get_corresponding_pathes_dirs
from wsipack.utils.flexparse import FlexArgumentParser
import numpy as np

from .wsd_image import ImageReader, write_array


def resize_mask(mask_path, wsi_path, out_path, spacing, overwrite=False, decr=False, spacing_tolerance=0.3):
    if str(out_path)==str(mask_path) or str(out_path)==str(wsi_path):
        raise ValueError('out_path %s may not overlap with mask or slide' % str(out_path))
    if not overwrite and Path(out_path).exists():
        print('not overwriting %s' % str(out_path))
        return

    if not Path(mask_path).exists():
        raise ValueError('mask %s doesnt exist' % mask_path)
    if not Path(wsi_path).exists():
        raise ValueError('slide %s doesnt exist' % wsi_path)


    print('processing %s' % str(mask_path))
    mreader = ImageReader(mask_path, spacing_tolerance=spacing_tolerance)
    try:
        found_spacing = mreader.refine(spacing)
        print('spacing %.2f already present in mask %s, skipping' % (spacing, str(mask_path)))
        if str(mask_path)!=str(out_path):
            print('copying %s' % str(out_path))
            shutil.copyfile(str(mask_path), str(out_path))
    except:
        #spacing not present - resize
        reader = ImageReader(wsi_path, spacing_tolerance=spacing_tolerance)
        spacing = reader.refine(spacing)
        lev = reader.level(spacing)
        shape = reader.shapes[lev]
        reader.close()

        mask = mreader.content(mreader.spacings[0])
        mreader.close()

        if decr:
            print('decrementing mask with vals %s' % str(np.unique(mask)))
            mask -= 1
            mask[mask<0] = 0

        mask = cv2.resize(mask, (shape[1], shape[0]))
        # from dptshared.cool_utils import showim
        # showim(mask)
        print('writing %s...' % str(out_path))
        write_array(mask, out_path, spacing)

def resize_masks(mask_dir, wsi_dir, out_dir, spacing, suffix='tif', overwrite=False, **kwargs):
    if ',' in suffix:
        suffix = suffix.split(',')
    wsi_pathes, mask_pathes = get_corresponding_pathes_dirs(wsi_dir, mask_dir, must_all_match=False,
                                                            take_shortest=True, ending1=suffix)
    if None in mask_pathes:
        print('%d wsis, but only %d masks' % (len(wsi_pathes), len([mp for mp in mask_pathes if mp is not None])))
        wsi_pathes, mask_pathes = get_corresponding_pathes_dirs(wsi_dir, mask_dir, must_all_match=False,
                                                ignore_missing=True, ending1=suffix, take_shortest=True)
    print('processing %d masks' % len(mask_pathes))
    out_dir = Path(out_dir)
    ensure_dir_exists(out_dir)
    for wp, mp in tqdm(zip(wsi_pathes, mask_pathes)):
        out_path = out_dir/Path(mp).name
        resize_mask(mp, wp, out_path=out_path, spacing=spacing, overwrite=overwrite, **kwargs)
    print('Done!')



def main():
    parser = FlexArgumentParser()
    parser.add_argument('--mask_dir',  required=True, type=str)
    parser.add_argument('--wsi_dir',   required=True, type=str)
    parser.add_argument('--out_dir',   required=True, type=str)
    parser.add_argument('--suffix',   required=False, type=str, default='tif,svs,ndpi,mrxs')
    parser.add_argument('--spacing',   required=True, type=float)
    parser.add_argument('-w', '--overwrite',  action='store_true', )
    args = parser.parse_args()
    print('args:', args)
    resize_masks(**args)

if __name__ == '__main__':
    main()