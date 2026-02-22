import argparse
import getpass
from pathlib import Path

import pandas as pd

from wsipack.utils.cool_utils import can_open_file, ensure_dir_exists, write_yaml_dict
from wsipack.utils.df_utils import print_df, df_save
from wsipack.utils.files_utils import FilesInfo
from wsipack.utils.path_utils import PathUtils
from wsipack.utils.cool_utils import df_row_to_dict, get_dir_size, get_file_size
from wsipack.wsi.contour_utils import px_to_mm2
from wsipack.wsi.wsi_read import ImageReader, create_reader


class SlidesInfo(FilesInfo):
    def __init__(self, files_dir):
        super().__init__(files_dir, not_containing=['csv','txt','xlsx', 'xml','yaml', 'json', 'txt'])

    def _get_infos_path(self, cdir):
        return Path(cdir)/'slide_infos.csv'

def create_slides_info_summary(slide_dir, out_dir=None, overwrite=False, spacing_tolerance=0.25,
                               not_containing=['.txt', '.csv', '.xml', '.yaml', '.json', 'xlsx', '.ini']):
    slide_dir = Path(slide_dir)
    if out_dir is None:
        out_dir = slide_dir
    out_dir = Path(out_dir)

    print('creating summary for %s' % slide_dir)
    detail_out_path = out_dir/'slide_infos.csv'
    summary_out_path = out_dir/'slide_summary.yaml'
    if not overwrite and can_open_file(detail_out_path, ls_parent=True):
        print('skipping already existing %s' % detail_out_path)
        return

    failed = []
    entries = []
    slides = PathUtils.list_pathes(slide_dir, type='file', sort=True, not_containing=not_containing)
    print('summarizing %d slides' % len(slides))
    for i,wsi_path in enumerate(slides):
        print('.', end='')
        if i%100==0: print('%d/%d' % (i, len(slides)))
        name = wsi_path.stem
        try:
            reader = create_reader(str(wsi_path), spacing_tolerance=spacing_tolerance)
            if reader.spacings[0] is None:
                print('skipping slide without spacings %s' % str(wsi_path))
                failed.append(wsi_path)
                continue
        except Exception as ex:
            print(ex)
            print('skipping failed %s' % wsi_path)
            failed.append(wsi_path)
            continue

        if 'mrxs' in wsi_path.suffix:
            file_size = get_dir_size(str(wsi_path).replace('.mrxs',''))
        else:
            file_size = get_file_size(wsi_path, mb=True)

        has_sp_map = {}
        for sp in [0.25, 0.5, 1, 2, 4, 8]:
            has_sp_str = 'has_sp%.2f' % sp
            has_sp_str = has_sp_str.rstrip('0')
            has_sp_str = has_sp_str.replace('.', '')
            try:
                reader.refine(sp)
                has_sp_map[has_sp_str] = True
            except:
                has_sp_map[has_sp_str] = False
        width = reader.shapes[0][1]
        height = reader.shapes[0][0]
        area = int(px_to_mm2(width * height, spacing=reader.spacings[0]))

        entries.append(dict(name=str(name), spacing=reader.spacings[0], width=width, height=height,
                     n_levels=len(reader.spacings), spacings=str(reader.spacings),
                            spacing_min=min(reader.spacings), spacing_max=max(reader.spacings),
                            shape=reader.shapes[0], shapes=str(reader.shapes),
                     filesize=file_size, area=area, **has_sp_map))
        reader.close()

    if len(entries)>0:
        df = pd.DataFrame(entries) #setting dtype here doesnt work
        df['name'] = df['name'].astype(str)
        if str(out_dir)!=str(slide_dir):
            ensure_dir_exists(out_dir)
        df_save(df, detail_out_path, overwrite=overwrite)
        df_save(df, str(detail_out_path).replace('.csv','.xlsx'), overwrite=overwrite)

        shape_min = df_row_to_dict(df[df.area==df.area.min()])['shape']
        shape_max = df_row_to_dict(df[df.area==df.area.max()])['shape']
        summary = dict(n=len(entries),
                       filesize_min=int(df.filesize.min()), filesize_max=int(df.filesize.max()), filesize_sum=int(df.filesize.sum()),
                       area_min=int(df.area.min()), area_max=int(df.area.max()),
                       shape_min=str(shape_min), shape_max=str(shape_max))
        for hsp in has_sp_map.keys():
            summary[hsp.replace('has','n')] = int(df[hsp].sum())
        print('writing summary for %s:' % slide_dir)
        write_yaml_dict(summary_out_path, summary, overwrite=overwrite)
        print(summary)
        print(detail_out_path)
    else:
        print('no valid entries')

    if len(failed)>0:
        failed = [str(f) for f in failed]
        print('%d failed: %s:' % (len(failed), str(failed)))
        print('rm commands for failed slides')
        for f in failed:
            print('rm %s' % str(f))

if __name__ == '__main__':

    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('--slide_dir', required=True, type=str, help='slide dir')
    argument_parser.add_argument('--out_dir', required=False, type=str, help='if not specified uses the slide_dir')
    argument_parser.add_argument('-w', '--overwrite',   action='store_true')

    arguments = vars(argument_parser.parse_args())
    create_slides_info_summary(**arguments)