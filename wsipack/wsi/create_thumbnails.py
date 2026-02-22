from pathlib import Path
from tqdm import tqdm
import sys
from wsipack.utils.cool_utils import mkdir
from wsipack.utils.flexparse import FlexArgumentParser
from wsipack.utils.path_utils import PathUtils
from wsipack.wsi.wsi_utils import create_thumbnail

def create_thumbnails(wsi_dir, out_dir=None, overwrite=False, openslide=False,
                      **kwargs):
    pathes = PathUtils.list_pathes(wsi_dir, ending=['tif', 'tiff', 'mrxs', 'svs', 'npi', 'scn', 'bif', 'ndpi', 'dicom'])
    print('creating thumbnails for %d files in %s' % (len(pathes), wsi_dir))
    if out_dir is None:
        out_dir = str(Path(wsi_dir))+'_thumbnails'
    print('out_dir:', out_dir)

    mkdir(out_dir)
    failures = []; skipped = []
    counter=0
    for path in tqdm(pathes):
        wsi_name = path.name.split('.')[0]
        out_path = Path(out_dir) / (wsi_name + '.jpg')
        if out_path.exists() and not overwrite:
            print('skip existing %s' % str(out_path))
            skipped.append(path)
            continue
        success = create_thumbnail(path, out_path, overwrite=overwrite, openslide=openslide, **kwargs)
        counter+=success
        if not success:
            failures.append(path)
    if len(failures)>0:
        print('Failed for %d:' % (len(failures)))
        for f in failures:
            print(str(f))
    if len(skipped)>0:
        print('skipped %d' % len(skipped))
    print('%d thumbnails created!' % counter)


def my_main():
    # ensure_dir_exists(Path(out_path).parent)
    # create_thumbnail(path, out_path)

    path = "/data/pathology/archives/gastrointestinal/examode_biopsies_radboudumc_colon/reader_study_sel_addon"

    path = "/data/pathology/archives/gastrointestinal/examode_biopsies_radboudumc_colon/images_tif"
    path = "/home/witali/projects2/wsipack/documentation/assets"
    create_thumbnails(path)


if __name__ == '__main__':
    if len(sys.argv)==1:
        my_main()
    else:
        parser = FlexArgumentParser()
        parser.add_argument('--wsi_dir', required=True, type=str)
        parser.add_argument('--out_dir', required=False, default=None, type=str)
        parser.add_argument('--openslide', action='store_true')
        parser.add_argument('--overwrite', action='store_true')
        args = parser.parse_args()
        create_thumbnails(**args)