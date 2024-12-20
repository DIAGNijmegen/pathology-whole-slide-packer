import shutil

from pathlib import Path

from wsipack.create_tissue_masks import create_tissue_masks
from wsipack.pack_slides import pack_slide
from wsipack.utils.cool_utils import is_float, read_lines, mkdir
from wsipack.utils.flexparse import FlexArgumentParser
from wsipack.utils.path_utils import PathUtils, get_corresponding_pathes


def main(slide_dir, spacing, mask_dir=None, out_dir='/output/images/packed',
         mask_out_dir='/output/images/tissue-mask', tmp_dir='/tmp'):
    level=None
    try:
        if str(spacing).endswith('json'):
            # spacing_path = Path(input_dir)/'spacing.json'
            sp = read_lines(spacing)[0]
            print('spacing %s' % sp)
            spacing = float(sp)
        else:
            spacing = float(spacing)
    except:
        print(f'exception when parsing spacing {spacing}, using level 0')
        spacing = None
        level = 0
    if is_float(spacing) and spacing <= 0:
        print(f'invalid spacing {spacing}, using level 0')
        spacing = None
        level = 0
    slides = PathUtils.list_pathes(slide_dir, ending=['tif','svs','mrxs','ndpi','tiff','dcm'])
    print('%d slides: %s' % (len(slides), slides))
    if len(slides) == 0:
        raise ValueError('no slides found in %s' % slide_dir)

    if mask_dir is not None and Path(mask_dir).exists():
        mask_pathes = PathUtils.list_pathes(mask_dir, ending='tif')
        print('found %d masks' % len(mask_pathes))
        if len(mask_pathes) != len(slides):
            print('number of masks doesnt match number of slides')
            mask_pathes = None
    else:
        print('no masks found')
        mask_pathes = None
    if mask_pathes is None:
        mask_dir = tmp_dir+'/tissue_masks'
        print('creating tissue masks for %s in %s' % (slide_dir, mask_dir))
        create_tissue_masks(slide_dir, mask_dir, spacing=spacing, level=level,
                            wsi_suffix=['tif', 'tiff', 'svs', 'ndpi', 'mrxs'])
        mask_pathes = PathUtils.list_pathes(mask_dir, ending='tif')

    if len(slides)==1 and len(mask_pathes)==1:
        masks = mask_pathes
    else:
        slides, masks = get_corresponding_pathes(slides, mask_pathes, must_all_match=True, take_shortest=True)

    pack_slide(slides, masks, out_dir=tmp_dir, spacing=spacing, level=level)
    packed_slides_dir = Path(tmp_dir)/'packed'
    created_packed = PathUtils.list_pathes(packed_slides_dir, ending='tif')
    print('copying %d packed from %s to %s' % (len(created_packed), packed_slides_dir, out_dir))
    mkdir(out_dir)
    for p in created_packed:
        shutil.copyfile(p, Path(out_dir)/p.name)
    created_masks = PathUtils.list_pathes(Path(tmp_dir)/'tissue_masks', ending='tif')
    print('copying %d masks to %s' % (len(created_masks), mask_out_dir))
    mkdir(mask_out_dir)
    for p in created_masks:
        shutil.copyfile(p, Path(mask_out_dir)/p.name)
    print('Done!')

def _example():

    main(slide_dir='../documentation/assets', spacing=2, mask_dir='./out/tissue_masks', out_dir='./out/packed',
         mask_out_dir='./out/tissue-mask', tmp_dir='./out/tmp')

if __name__ == '__main__':
    parser = FlexArgumentParser()
    args = parser.parse_args()
    print('args:', args)
    main(**args)
    # _example()