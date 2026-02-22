import shutil
from pathlib import Path
from tqdm import tqdm

from wsipack.utils.io_utils import move_file, copy_from_to


def px_to_um2(px, spacing):
    return px*(spacing**2)


def move_slide(old_path, new_path, dry_run=False, **kwargs):
    old_path = Path(old_path)
    new_path = Path(new_path)

    if 'mrxs' in old_path.suffix:
        old_folder = old_path.parent/old_path.stem
        new_folder = new_path.parent/new_path.stem
        move_file(old_folder, new_folder, dry_run=dry_run, **kwargs)
    move_file(old_path, new_path, dry_run=dry_run, **kwargs)


def delete_slide(path):
    print('deleting %s' % str(path))
    path = Path(path)
    if path.exists():
        path.unlink()

    if path.suffix in ['.mrxs', 'mrxs']:
        dirpath = str(path).replace('.mrxs','')
        if Path(dirpath).exists():
            print('deleting %s' % str(dirpath))
            shutil.rmtree(path=dirpath)


def copy_slide_to(path, out_dir):
    out_path = Path(out_dir)/Path(path).name
    return copy_slide(path, out_path)


def copy_slide(path, out_path):
    copy_from_to(path, out_path)
    if str(path).endswith('.mrxs'):
        slide_dir = str(path)[:-5]
        out_dir_path = Path(out_path).parent/Path(slide_dir).name
        copy_from_to(slide_dir, out_dir_path)
    return out_path


def move_slides_map(dir_ending_map, rename_map, ignore_missing=False, dry_run=False, verbose=False, **kwargs):
    """ dir_ending_map: {img_dir:suffix}, rename_map: {old:new}  """
    missing = []
    for img_dir, ending in dir_ending_map.items():
        print('renaming in %s' % img_dir)
        for old_name, new_name in rename_map.items():
            old_path = Path(img_dir)/(old_name+ending)
            new_path = Path(img_dir)/(new_name+ending)
            if not old_path.exists():
                missing.append(old_path)
            else:
                move_slide(old_path, new_path, dry_run=dry_run, ignore_missing=ignore_missing, **kwargs)
    print('%d missing files' % len(missing))
    if len(missing)>0 and verbose:
        print(missing)
    return missing


def move_slides(old_pathes, new_pathes, dry_run=False, **kwargs):
    print('moving %d slides' % len(old_pathes))
    if len(old_pathes)!=len(new_pathes):
        raise ValueError('different number of pathes: %d old, but %d new' % (len(old_pathes), len(new_pathes)))
    for op, np in tqdm(zip(old_pathes, new_pathes)):
        move_slide(op, np, dry_run=dry_run, **kwargs)
