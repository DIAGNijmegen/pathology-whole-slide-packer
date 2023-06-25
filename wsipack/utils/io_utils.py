import shutil, os
import subprocess
from pathlib import Path

from tqdm import tqdm

from wsipack.utils.cool_utils import *
from wsipack.utils.path_utils import PathUtils


def subprocess_cmd(command, verbose=False):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    print('runnning cmd %s' % command)
    proc_stdout = process.communicate()[0].strip()
    if verbose:
        print(proc_stdout)

def rsync_files(src, dst, verbose=False, compress=False):
    with timer('rsync %s' % str(src)):
        flags = 'avh'
        if compress: flags+='z'
        return subprocess_cmd('rsync -%s %s %s' % (flags, str(src), str(dst)), verbose=verbose)

def rsync_dir(src, dst, verbose=False):
    ensure_dir_exists(dst)
    src = str(Path(src).absolute())+'/'
    dst = str(Path(dst).absolute())+'/'
    rsync_files(src, dst, verbose=verbose)

def rsync_dirs(src_dirs, tar_dirs, verbose=False):
    for src, tar in zip(src_dirs, tar_dirs):
        rsync_dir(src, tar, verbose=verbose)

def cache_dirs(src_dirs, tar_dir, name_depth=1, name_flatten='auto'):
    """ rsyncs the src_dirs to the tar_dir keeping the directory structure.
     name_depth defines how many subdirs are created in tar_dir from the parent-structure of src_dir
     """
    print('caching %s to %s' % (str(src_dirs), str(tar_dir)), flush=True)
    if is_string(src_dirs) and ',' in src_dirs:
        src_dirs = src_dirs.split(',')

    iterable_out = True
    if not is_iterable(src_dirs):
        src_dirs = [src_dirs]
        iterable_out = False

    name_depth = max(name_depth, 1)
    tar_dirs = []
    for src_dir in src_dirs:
        parts = Path(src_dir).parts[-name_depth:]
        parts_len = sum([len(p) for p in parts])
        if (name_flatten=='auto' and parts_len<15) or parts_len==True:
            name = '_'.join(parts)
        else:
            name = '/'.join(parts)
        tar_dir_i = Path(tar_dir)/name
        tar_dirs.append(tar_dir_i)
    rsync_dirs(src_dirs, tar_dirs)
    if not iterable_out and len(tar_dirs)==1:
        return tar_dirs[0]
    return tar_dirs

def move_file(old_path, new_path, dry_run=False, ignore_missing=False, ssh_node=None):
    old_path = Path(old_path)
    new_path = Path(new_path)

    if not old_path.exists():
        if ignore_missing:
            print('ignore missing %s' % str(old_path))
            return 0
        else:
            raise ValueError('source %s doesnt exist' % str(old_path))

    if new_path.exists():
        raise ValueError('moving %s not possible, %s already exists' % (old_path.name, str(new_path)))

    if not new_path.parent.exists() and not dry_run:
        mkdir(new_path.parent)

    if ssh_node is None:
        if dry_run:
            print('not moving %s to %s' % (str(old_path), str(new_path)))
        else:
            shutil.move(str(old_path), str(new_path))
    else:
        print(run_cmd("ssh %s 'mv %s %s'" % (ssh_node, str(old_path), str(new_path)), dry_run=dry_run))
    return 1

def move_files(pathes, new_pathes, dry_run=False, delete_links=False):
    """ moves files without checking if they or their targets exist """
    if len(pathes)==0:
        print('nothing to move')
        return
    print('moving %d files...' % len(pathes))
    for path, new_path in tqdm(zip(pathes, new_pathes)):
        path = Path(path)
        new_path = Path(new_path)

        if delete_links and path.is_symlink():
            if dry_run:
                print('not unlinking %s' % str(path))
            else:
                path.unlink()
            continue

        if dry_run:
            if path.parent==new_path.parent:
                print(f'not renaming {path.name} to {new_path.name} in {new_path.parent}')
                continue
            else:
                print('not moving %s to %s' % (str(path), str(new_path)))
        else:
            shutil.move(str(path), str(new_path))
    print('moved %d files' % len(pathes))

def move_files_names(src_dir, tar_dir, names, suffix, dry_run=False, ssh_node=None, **kwargs):
    mkdir(tar_dir)
    missing = []
    for name in tqdm(names):
        src_path = Path(src_dir)/(name+suffix)
        tar_path = Path(tar_dir)/(name+suffix)
        ok = move_file(src_path, tar_path, ssh_node=ssh_node, dry_run=dry_run, **kwargs)
        if not ok:
            missing.append(src_path)

    print('%d missing' % len(missing))
    print('Done')

def compare_directories_names(dir1, dir2):
    files1 = PathUtils.list_pathes(dir1, ret='name', sort=True)
    files2 = PathUtils.list_pathes(dir2, ret='name', sort=True)
    inters = list_intersection(files1, files2)
    not_in2 = list_not_in(files1, files2)
    print('%d in %s, %d in %s, %d intersection' % (len(files1), dir1, len(files2), dir2, len(inters)))
    print('%d in %s not in %s:' % (len(not_in2), dir1, dir2))
    print(not_in2)


def copy_files_to_dir(pathes, target_dir, exclude=None, dry_run=False, replace_map={}, overwrite=False):
    """ pathes: list of pathes or the path of a file with these pathes as lines"""
    exclude_names = []
    if exclude is not None:
        if is_string(exclude) and Path(exclude).is_dir():
            exclude_pathes = PathUtils.list_pathes(exclude)
        elif is_string(exclude) and Path(exclude).is_file():
            exclude_pathes = read_lines(exclude)
        elif is_iterable(exclude):
            exclude_pathes = exclude
        else: raise ValueError('unknown exclude %s' % exclude)
        exclude_names = [Path(ep).stem for ep in exclude_pathes]
    if is_string(pathes):
        pathes = read_lines(pathes)
    ensure_dir_exists(target_dir)
    skipped = []; copied = []
    for path in tqdm(pathes):
        path = Path(path)
        name = path.name
        for k,v in replace_map.items():
            name = name.replace(k,v)
        target_path = Path(target_dir)/name
        if path.stem in exclude_names:
            print('skipping excluded %s' % path)
        elif target_path.exists() and not overwrite:
            print('skipping %s' % path)
            skipped.append(path)
        else:
            print('copying %s to %s' % (str(path),str(target_path)))
            copied.append(path)
            if not dry_run:
                shutil.copyfile(str(path), target_path)
    print('%d copied, %d skipped' % (len(copied), len(skipped)))
    if dry_run: print('dry_run!')

def delete_files(pathes, dry_run=False):
    print('deleting %d files' % len(pathes))
    for p in tqdm(pathes):
        if dry_run:
            print('not deleting %s' % str(p))
            continue
        p = Path(p)
        if p.exists():
            if p.is_symlink():
                p.unlink(missing_ok=True)
            elif p.is_dir():
                shutil.rmtree(str(p))
            else:
                os.remove(str(p))

def delete_files_containing(folder, containings, dry_run=False, depth=0, ssh_node=None):
    print('deleting files containing', containings)
    files = PathUtils.list_pathes(folder, ret='str', depth=depth, ssh_node=ssh_node)
    to_delete = []
    for f in files:
        for word in containings:
            if word in f:
                to_delete.append(f)
                break

    delete_files(to_delete, dry_run=dry_run)

def find_files(dir, pattern, ssh_node=None, dry_run=True):
    cmd = 'find %s/%s' % (dir, pattern)
    if ssh_node is not None:
        cmd = f'ssh {ssh_node} {cmd}'

    run_cmd_live(cmd, dry_run=dry_run)


def copy_from_to(source, target):
    if target is None:
        return str(source)
    target = Path(target)
    if source is None:
        return str(target)
    source = Path(source)

    try:
        if target.exists():
            print('skipping copying %s to %s, since target exists' % (source, target))
        elif not source.exists():
            print("can't copy source %s since it doesn't exist" % source)
        else:
            if source.is_dir():
                print('Copying dir %s to %s' % (source, target), flush=True)
                shutil.copytree(source, target)
            else:
                print('Copying file %s to %s' % (source, target), flush=True)
                shutil.copyfile(source, target)
        return str(target)
    except Exception as e:
        print('Failed to copy file {file}. Using original location instead. Exception: {e}' \
              .format(file=str(source),e=e),flush=True)
        return str(source)