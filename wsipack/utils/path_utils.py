import glob
import os
import subprocess
from pathlib import Path

from wsipack.utils.cool_utils import is_string, is_iterable, list_not_in
from tqdm import tqdm

class PathUtils(object):
    @staticmethod
    def filter_containing_or(names, containing_or=None, split_commas=True):
        selection = names
        if containing_or is None:
            containing_or = []
        elif is_string(containing_or):
            if ',' in containing_or and split_commas:
                containing_or = containing_or.split(',')
            else:
                containing_or = [containing_or]
        if len(containing_or)>0:
            selection = []
            for name in names:
                for incl in containing_or:
                    if incl in name:
                        selection.append(name)
                        break

        return selection

    @staticmethod
    def filter_containing_and(names, containing_and=None, split_commas=True):
        selection = names
        if containing_and is None:
            containing_and = []
        elif is_string(containing_and):
            if ',' in containing_and and split_commas:
                containing_and = containing_and.split(',')
            else:
                containing_and = [containing_and]
        if len(containing_and)>0:
            selection = []
            for name in names:
                contains_all = True
                for incl in containing_and:
                    if incl not in name:
                        contains_all = False
                        break
                if contains_all:
                    selection.append(name)

        return selection

    @staticmethod
    def filter_not_containing(names, not_containing=None, split_commas=True):
        selection = names
        if not_containing is None:
            not_containing = []
        elif is_string(not_containing):
            if split_commas:
                not_containing = not_containing.split(',')
            else:
                not_containing = [not_containing]
        if len(not_containing)>0:
            selection = []
            for name in names:
                skip = False
                for not_cont in not_containing:
                    if not_cont in str(name):
                        skip = True
                        break
                if not skip:
                    selection.append(name)

        return selection

    @staticmethod
    def filter_ending(names, ending=None):
        if ending is None:
            return names
        if len(ending) == 1 and ending[0] == None:
            return names

        if not is_iterable(ending):
            ending = [ending]
        selection = []
        for end in ending:
            if end and len(end) > 0:
                selection.extend([name for name in names if name.endswith(end)])
        selection = list(set(selection))
        return selection

    @staticmethod
    def filter_type(root, names, type='all'):
        if type=='file':
            names = [name for name in names if os.path.isfile(str(Path(root)/name))]
        elif type=='dir':
            names = [name for name in names if os.path.isdir(str(Path(root)/name))]
        elif type!='all':
            raise ValueError('unknown type %s' % type)
        return names

    @staticmethod
    def list_files(*args, **kwargs):
        return PathUtils.list_pathes(*args, **kwargs, type='file')

    @staticmethod
    def _parse_roots(roots):
        if not is_iterable(roots):
            roots = str(roots)
            if ',' in roots and not Path(roots).exists():
                roots = roots.split(',')
            else:
                roots = [roots]
        return roots

    @staticmethod
    def list_slides(roots, ending=['svs','tif','mrxs', 'ndpi', 'tiff'], **kwargs):
        return PathUtils.list_pathes(roots, ending=ending, **kwargs)

    @staticmethod
    def list_pathes(roots, containing=None, containing_or=None, containing_and=None, not_containing=None,
                    ending=None, ret='path', sort=False, ignore_dots=True,
                    type='all', split_commas=True, ssh_node=None, depth=1):
        """ type: all, file, dir
        ret: ['path', 'str', 'name', 'stem'] """
        roots = PathUtils._parse_roots(roots)

        pathes = []
        for root in roots:
            if '*' in str(root):
                root_dir = Path(root).parent
            else:
                root_dir = root
            ret_values = ['path', 'str', 'name', 'stem']
            if ret not in ret_values:
                raise ValueError('ret must be in %s' % str(ret_values))

            # if '*' in root:
            #     root = Path(root).parent

            if ssh_node is None or ssh_node==False:
                if '*' in str(root):
                    names = [Path(p).name for p in glob.glob(str(root))]
                else:
                    names = os.listdir(str(root))
                names = PathUtils.filter_type(root, names, type)
            else:
                if not is_string(ssh_node): ssh_node = 'ditto'
                names = PathUtils._list_dir_ssh(root, ssh_node=ssh_node, type=type, ret='name')

            names = PathUtils.filter_ending(names, ending)
            names = PathUtils.filter_not_containing(names, not_containing)
            names = PathUtils.filter_containing_or(names, containing_or, split_commas=split_commas)
            names = PathUtils.filter_containing_and(names, containing, split_commas=split_commas)
            names = PathUtils.filter_containing_and(names, containing_and, split_commas=split_commas)
            for name in names:
                if ignore_dots and name.startswith('.'):
                    continue

                path = Path(root_dir)/name
                if ret=='path':
                    pathes.append(path)
                elif ret == 'str':
                    pathes.append(str(path))
                elif ret == 'name':
                    pathes.append(path.name)
                elif ret == 'stem':
                    pathes.append(path.stem)

        if depth>1:
            subdirs = PathUtils.list_pathes(roots, type='dir', ssh_node=ssh_node)
            for subdir in subdirs:
                subpathes = PathUtils.list_pathes(subdir, containing=containing, containing_or=containing_or, containing_and=containing_and,
                                                  not_containing=not_containing, ending=ending, ret=ret, ignore_dots=ignore_dots,
                                                  type=type, split_commas=split_commas, depth=depth-1, ssh_node=ssh_node)
                pathes.extend(subpathes)
        if sort:
            pathes = sorted(pathes)
        return pathes

    @staticmethod
    def _list_dir_ssh(rdir, ssh_node='ditto', type=None, ret='path', min_depth=1, depth=1, ending=None):
        pathes = []
        if ending is not None:
            name_str = ' -name *'+ending
        else:
            name_str = ''
        cmd = f"ssh {ssh_node} find {rdir} -mindepth {min_depth} -maxdepth {depth}{name_str}"
        if type is None or str(type).lower() in ['none', 'all', '']:
            pass
        elif type == 'file' or type == 'f':
            cmd += ' -type f'
        elif type == 'dir' or type == 'd':
            cmd += ' -type d'
        else:
            raise ValueError('unknown type %s' % type)
        # cmd = f"ssh {ssh_node} ls %s" % rdir
        lines = subprocess.getoutput(cmd).splitlines()
        # print('%d lines' % len(lines))
        lines = [l for l in lines if l not in ['', '.']]

        if ret == 'path':
            lines = [Path(l) for l in lines]
        elif ret == 'str':
            pass
        elif ret == 'name':
            lines = [Path(l).name for l in lines]
        elif ret == 'stem':
            lines = [Path(l).stem for l in lines]
        pathes.extend(lines)
        return pathes


    @staticmethod
    def list_dir_ssh(roots, ssh_node='ditto', type=None, ret='path', depth=1, **kwargs):
        roots = PathUtils._parse_roots(roots)
        pathes = []
        for rdir in roots:
            pathes.extend(PathUtils._list_dir_ssh(rdir, ssh_node=ssh_node, type=type, ret=ret, depth=depth, **kwargs))
        return pathes


def get_stem(path):
    return str(path).split('/')[-1].split('.')[0]


def get_pathes_starting_with(name, pathes):
    """
    :param name: filename-prefix
    :param pathes: list of pathes
    :return: pathes which name starts with 'name'
    """
    found = []
    for p in pathes:
        if p.name.startswith(name):
            found.append(p)
    return found

def pathes_ending_with(suffix, pathes):
    """
    :param name: filename-prefix
    :param pathes: list of pathes
    :return: pathes which name ends with 'name'
    """
    found = []
    for p in pathes:
        if p.name.endswith(suffix):
            found.append(p)
    return found

def get_path_named_like_in_dir(name, search_dir, recursvie=False, ending=None, take_shortest=False):
    """ looks for a file in search_dir, which starts with 'name' """
    if search_dir is None:
        return None
    all_pathes = PathUtils.list_pathes(search_dir, ending=ending, recursvie=recursvie)
    return get_path_named_like(name, all_pathes, take_shortest=take_shortest)

def get_path_named_like(name, all_pathes, same_name=False, take_shortest=False, as_string=False,
                        replacements={}):
    """ assumes there are no two names where one name contains the other one
     replacements: dict {repl_str:with_str} replaces repl_stri with the with_str for matching the pathes """
    # same_ending = None
    # if ignore_same_ending is not None:
    #     same_ending = all_pathes[0].suffix
    #     if ignore_same_ending:
    #         same_ending = all_pathes[0].name.split('_')[-1]
    #         for p in all_pathes:
    #             ending = p.name.split('_')[-1]
    #             if ending!=same_ending:
    #                 same_ending = None #files dont have the same ending
    #                 break
    #         if same_ending is not None and len(all_pathes)>1:
    #             same_ending = '_'+same_ending

    if isinstance(name, Path):
        name = name.stem

    found = []
    for p in all_pathes:
        p = Path(p)
        other = p.stem
        for k,v in replacements.items():
            other = other.replace(k,v)
        # if same_ending is not None:
        #     pname = p.name.replace(same_ending, '')
        if same_name:
            if other == name:
                found.append(p)
        else:
            if other.startswith(name):# or name.startswith(p.stem):
                found.append(p)
    if found is None or len(found)==0:
        return None
    if len(found)>1:
        found.sort(key=lambda s: len(str(s)))
        if take_shortest:
            print('selecting %s to match %s from %d found pathes %s' % (found[0], name, len(found), str([fo.stem for fo in found[1:]])))
        else:
            raise ValueError('too many files found for %s, :%s' % (name, str(found)))
    result = found[0]
    if as_string:
        result = str(result)
    return result

def get_matching_pathes(pathes1, pathes2, must_all_match=False, **kwargs):
    return get_corresponding_pathes(pathes1, pathes2, must_all_match=must_all_match, **kwargs)

def get_corresponding_pathes(pathes1, pathes2, must_all_match=False, ignore_missing=False,
                             as_string=False, ignore_missing2=True, **kwargs):
    """ finds pathes in pathes2 named like in pathes1 and returns both matching pathes
    ignore_missing=True, returns only matches, otherwise returns None for missing p2
    must_all_match: raises error if doesnt find match
    """
    sel1 = []; sel2 = []
    for p1 in tqdm(pathes1):
        p2 = get_path_named_like(Path(p1).stem, pathes2, as_string=as_string, **kwargs)
        if p2 is None:
            if must_all_match:
                raise ValueError('no match for %s in %s' % (Path(p1).stem, str(pathes2)))
            elif ignore_missing:
                continue
            else:
                pass
                #p2 will be added as None
        if as_string:
            p1 = str(p1)
            if p2 is not None:
                p2 = str(p2)
        sel1.append(p1)
        sel2.append(p2)
    if not ignore_missing2:
        s2 = list(set([str(p) for p in sel2]))
        a2 = list(set([str(p) for p in pathes2]))
        m2 = list_not_in(a2, s2)
        if len(s2)!=len(pathes2):
            print('no matches for %d %s' % (len(m2), m2))
    return sel1, sel2


def get_corresponding_pathes_all(pathes1, pathes2, **kwargs):
    _, found = get_corresponding_pathes(pathes1, pathes2, must_all_match=True, **kwargs)
    return found

def get_corresponding_pathes_dirs(dir1, dir2, containing1=None, ending1=None, not_containing1=None,
                                  containing2=None, ending2=None, not_containing2=None,
                                  take_shortest=False, must_all_match=False, allow_dirs=False,
                                  ignore_missing=False, **kwargs):
    """ dir: directory or list of pathes"""
    type = 'file'
    if allow_dirs:
        type = 'all'
    if is_iterable(dir1):
        pathes1 = dir1
        if ending1 is not None: raise ValueError('no ending if list of pathes1 is given')
    else:
        pathes1 = PathUtils.list_pathes(dir1, containing_or=containing1, ending=ending1,
                                        not_containing=not_containing1, type=type)
    if is_iterable(dir2):
        pathes2 = dir2
        if ending2 is not None: raise ValueError('no ending if list of pathes2 is given')
    else:
        pathes2 = PathUtils.list_pathes(dir2, containing_or=containing2, ending=ending2,
                                        not_containing=not_containing2, type=type)

    return get_corresponding_pathes(pathes1, pathes2, take_shortest=take_shortest,
                                    must_all_match=must_all_match,
                                    ignore_missing=ignore_missing, **kwargs)

def match_pathes_in_dirs(source_dir, match_dir, source_ending=None, match_ending=None, source_replace={}):
    """ source_replace: replace in source name key with val before matching """
    if is_iterable(source_dir):
        source_pathes = source_dir
    else:
        source_pathes = PathUtils.list_pathes(source_dir, ending=source_ending)
    if is_iterable(match_dir):
        to_match_pathes = match_dir
    else:
        to_match_pathes = PathUtils.list_pathes(match_dir, ending=match_ending)
    found_sources = []; found_matches = []; missing = []
    for sp in source_pathes:
        sp_name = sp.stem
        for se,re in source_replace.items():
            sp_name = sp_name.replace(se, re)
        matched_path = get_path_named_like(sp_name, to_match_pathes)
        if matched_path is not None:
            found_sources.append(str(sp))
            found_matches.append(str(matched_path))
        else:
            missing.append(str(sp))
    print('matched %d of %d files in %s to %s' % (len(found_sources), len(source_pathes), str(source_dir), str(match_dir)))
    if len(missing)>0:
        print('%d missed matches: %s' % (len(missing), str(missing)))
    return found_sources, found_matches


