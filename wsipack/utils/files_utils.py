import shutil, os, sys
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import pandas as pd
import os
import shutil

from wsipack.utils.cool_utils import is_iterable, is_string, can_open_file, mkdir
from wsipack.utils.path_utils import PathUtils


class FilesInfo(object):
    name_col = 'name' #name of the slide without suffix" (in the csv with suffix)
    suffix_col = 'suffix'
    path_col = 'path' #full path

    def _get_infos_path(self, cdir):
        raise ValueError('implement: name of files info!')

    def __init__(self, files_dir, not_containing=None):
        if is_string(files_dir) and ',' in files_dir:
            files_dir = files_dir.split(',')
        if not is_iterable(files_dir):
            files_dir = [files_dir]
        self.compressed_dir = files_dir

        self.df = None
        for cdir in files_dir:
            if not Path(cdir).exists():
                print('skipping non-existing %s' % cdir)
                continue
            df = None
            infos_path = self._get_infos_path(cdir)
            if can_open_file(infos_path, ls_parent=True):
                df = pd.read_csv(str(infos_path))
                if self.suffix_col not in df:
                    names = df[self.name_col].values
                    suffixes = [Path(name).suffix for name in names]
                    df[self.suffix_col] = suffixes
                    names = [Path(name).stem for name in names]
                    df[self.name_col] = names
                names = df[self.name_col].values
                suffixes = df[self.suffix_col].values
                pathes = []
                for i in range(len(df)):
                    pathes.append(str(Path(cdir)/(names[i]+suffixes[i])))
                df[self.path_col] = pathes

            else:
                pathes = PathUtils.list_pathes(cdir, not_containing=not_containing, type='file')
                stems = [p.stem for p in pathes]
                suffixes = [p.suffix for p in pathes]
                pathes = [str(p) for p in pathes]
                if len(pathes)>0:
                    df = pd.DataFrame({self.name_col:stems, self.path_col:pathes, self.suffix_col:suffixes})
            if self.df is None:
                self.df = df
            elif df is not None:
                dfm = pd.concat([self.df, df], ignore_index=True, axis=0)
                n_merged =  dfm[self.name_col].nunique()
                if n_merged!=(self.df[self.name_col].nunique()+df[self.name_col].nunique()):
                    raise ValueError('non unique file names! %d merged, but should be %d' % \
                                     n_merged, self.df[self.name_col].nunique()+df[self.name_col].nunique())
                self.df = dfm

    def infos_exist(self):
        exist = [Path(self._get_infos_path(cdir)).exists() for cdir in self.compressed_dir]
        return all(exist)

    def get_df(self):
        return self.df

    def get_df_names(self):
        return self.df[self.name_col].copy()

    def write(self, df, out_path=None, with_path=False):
        if out_path is None and len(self.compressed_dir)!=1:
            raise ValueError('multiple compressed dirs, specify out_path explicitly')
        if out_path is None:
            out_path = self._get_infos_path(self.compressed_dir[0])
        if with_path:
            dfw = df
        else:
            dfw = df[[col for col in df if str(col)!=self.path_col]]
        dfw.to_csv(str(out_path), index=None)
        self.df = df
        print('Compressed infos created: %s' % Path(out_path).absolute())


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



def copy_files_from_list(file_list, src_dir, tar_dir, dry_run=False):
    with open(file_list, 'r') as file:
        file_names = file.read().splitlines()

    if not dry_run:
        mkdir(tar_dir)

    for file_name in tqdm(file_names):
        source_path = os.path.join(src_dir, file_name)
        destination_path = os.path.join(tar_dir, file_name)
        if not Path(source_path).exists():
            raise ValueError('%s missing' % source_path)
        if dry_run:
            print('not copying %s to %s' % (source_path, destination_path))
        else:
            shutil.copy2(source_path, destination_path)
            # print(f"File '{file_name}' copied successfully.")

def copy_files_from_lists(files_list_dir, src_dir, tar_dir, dry_run=False, **kwargs):
    files_lists = PathUtils.list_files(files_list_dir, ending='txt')
    for fl in files_lists:
        name = fl.stem
        # tar_dir = Path(tar_dir)/name
        copy_files_from_list(fl, src_dir=src_dir, tar_dir=tar_dir, dry_run=dry_run, **kwargs)
    print('Done')

def create_pathes_csv(dir, ending='tif'):
    pathes = PathUtils.list_pathes(dir, ending=ending, ret='str', sort=True)
    names = [Path(p).stem for p in pathes]
    df = pd.DataFrame(dict(name=names, path=pathes))
    out_path = Path(dir)/'pathes.csv'
    df.to_csv(str(out_path), index=False)
    print(str(out_path))