import importlib

import builtins

import datetime
import math
import multiprocessing
import sys, os

import time, json, shutil, string, subprocess
import timeit

import numpy
from tqdm import tqdm

import yaml
import random
import pandas as pd
import warnings

try:
    import StringIO
except ImportError:
    from io import StringIO

import numpy as np
from pathlib import Path

import collections
from collections import defaultdict
try:
    from collections import Callable
except: #for 3.10
    from collections.abc import Callable

from functools import partial
import matplotlib.pyplot as plt

from wsipack.utils.docker_utils import count_docker_cpus

def is_iterable(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, str)

def is_callable(obj):
    return hasattr(obj, '__call__')

def is_string(obj):
    return isinstance(obj, str)

def is_int(obj):
    return isinstance(obj, (int, np.integer))

def is_float(obj):
    return isinstance(obj, (float, np.floating))

def is_string_or_path(obj):
    return isinstance(obj, (str, Path))

def is_dict(obj):
    return isinstance(obj, dict)

def is_tuple(obj):
    return isinstance(obj, (tuple))

def is_list(obj):
    return isinstance(obj, list)

def is_list_or_tuple(obj):
    return isinstance(obj, (list, tuple))

def is_ndarray(img):
    return isinstance(img, np.ndarray)

def is_df(obj):
    return isinstance(obj, pd.DataFrame)

def is_empty(iterable):
    return iterable is None or len(iterable)==0

def str_remove_non_ascii(string):
    return string.encode('ascii', errors='ignore').decode()

def arr_wo_diag(A):
    """ removes the diagonal from the array """
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

def check_arr_inf_nan(print_all=True, **kwargs, ):
    for name,arr in kwargs.items():
        if np.isnan(arr).any():
            if print_all:
                print(arr)
            raise ValueError(f'{name} contain nan!')
        if not np.isfinite(arr).all():
            if print_all:
                print(arr)
            raise ValueError(f'{name} contain inf!')

def most_frequent(l, highest=False):
    """ returns the most frequent value in the iterable. if there are multiple values with same frequency, returns the
    lowest one (default sort cireterium), otherwise the highest
    """
    occurence_count = collections.Counter(l)
    counts = occurence_count.most_common()
    highest_count = counts[0][1]
    mcs = [(val,count) for (val,count) in counts if count==highest_count]
    values, counts = zip(*mcs) #these are the most common values
    assert len(set(counts))==1
    inds = sorted(range(len(values)), key=lambda k: values[k])
    if highest:
        return values[inds[-1]], counts[inds[-1]]
    else:
        return values[inds[0]], counts[inds[-1]]


def list_diff(l1, l2):
    """ returns list of values in l1, which are not in l2"""
    ldiff = list(set(l1) - set(l2))
    return ldiff

def lists_disjoint(l1, l2):
    return set(l1).isdisjoint(set(l2))

def list_intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def list_intersetion_rest(lst1, lst2, verbose=False, left_title='list1', right_title='list2'):
    inters = list_intersection(lst1, lst2)
    l1_surp = [v for v in lst1 if v not in inters]
    l2_surp = [v for v in lst2 if v not in inters]
    if verbose:
        print('%d inters, %d l1-surplus, %d l2-surplus' % (len(inters), len(lst1), len(lst2)))
        if len(inters): print('intersection:', str(inters))
        if len(l1_surp): print(left_title+' surplus:', str(l1_surp))
        if len(l2_surp): print(right_title+' surplus:', str(l2_surp))
    return inters, l1_surp, l2_surp

def list_not_in(lst1, lst2):
    """ returns values in lst1 not in lst2 """
    lst3 = [v for v in lst1 if v not in lst2]
    return lst3

def list_indexed(l, indizes):
    """ returns the items from the indizes """
    if indizes is None or len(indizes)==0:
        return []
    selection = [l[ind] for ind in indizes]
    return selection

def lists_indexed(indizes, *lists):
    indexed = [list_indexed(l, indizes) for l in lists]
    return tuple(indexed)

def list_where(list, indizes):
    """ gets the objects at the given indizes """
    l = []
    for ind in indizes:
        l.append(list[ind])
    return l

def lists_flatten(t):
    return [item for sublist in t for item in sublist]

def list_remove_duplicates(l):
    #should keep original item order
    return list(dict.fromkeys(l))

def list_strip(l, *args, **kwargs):
    return [str(v).strip(*args, **kwargs) for v in l]

def as_list(val):
    if not is_iterable(val):
        val = [val]
    return val

def as2d(arr):
    """ return a 2d-vector if 1d """
    if len(arr.shape)==1:
        arr = arr[:,None]
    return arr

def ind_where_not(iterable, indizes):
    all_ind = np.arange(len(iterable))
    diff_ind = np.setdiff1d(all_ind, np.array(indizes))
    return diff_ind

def list_where_not(list, indizes):
    diff_ind = ind_where_not(list, indizes)
    #     all_ind = np.arange(len(list))
    #     diff_ind = np.setdiff1d(all_ind, np.array(indizes))
    return list_where(list, diff_ind)

def ensure_dir_exists(path):
    if not path_exists(path):
        path = Path(path)
        return Path(path).mkdir(parents=True, exist_ok=True)
    return True

def mkdir(path):
    return ensure_dir_exists(path)

def mkdirs(*pathes):
    for path in pathes:
        mkdir(path)

def path_exists(path):
    return Path(path).exists()

def write_text(path, text, overwrite=True):
    if not overwrite and path_exists(path):
        print('not overwriting %s' % str(path))
    with open(str(path), 'w') as the_file:
        the_file.write(text)

def random_string(length=3, rs=None, digits=True, lowercase=True, uppercase=False):
    """ generates a random alphanumeric string"""
    if rs is None:
        rs = np.random.RandomState(int(time.time()))
    elif is_int(rs):
        rs = np.random.RandomState(rs)
    choice = ""
    if digits:
        choice+=string.digits
    if lowercase:
        choice+=string.ascii_lowercase
    if uppercase:
        choice+=string.ascii_uppercase
    if len(choice)==0: raise ValueError('not enouch choice for random_string!')
    stri = ''.join(rs.choice(np.array(list(choice))) for _ in range(length))
    return stri

def random_int_string(**kwargs):
    return random_string(lowercase=False, uppercase=False, digits=True, **kwargs)


def take_closest_number(l, number):
    if is_iterable(number): raise ValueError('first argument is list, second is the number')
    return min(l, key=lambda x: abs(x - number))

def take_closest_larger_number(l, number):
    filtered = [num for num in l if num >= number]
    if not filtered:
        return None
    return min(filtered)

def take_closest_number_index(l, number):
    closest = take_closest_number(l, number)
    for ind, val in enumerate(l):
        if val==closest: return ind
    return None

def take_closest_larger_number_index(l, number):
    closest = take_closest_larger_number(l, number)
    for ind, val in enumerate(l):
        if val==closest: return ind
    return None

class DictObject(object):
    def __init__(self, d, callback=None):
        self.__dict__ = d
        self.callback = callback

    def get_dict(self):
        return self.__dict__

    def __getitem__(self, key):
        if self.callback is not None:
            self.callback(key)
        return self.__dict__[key]

class Dict(dict):
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        self.__dict__ = self

#TODO:
#https://dev.to/0xbf/use-dot-syntax-to-access-dictionary-key-python-tips-10ec
class Dictd(dict):
    def __init__(self, d=None):
        if d is not None and len(d)>0:
            for k,v in d.items():
                self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

def dict_inverse(d):
    d_inv = {v: k for k, v in d.items()}
    return d_inv

def invert_dict_list(dic):
    """ Input: {key1:[value1, value2], key2:[value1]} Returns: {value1:[key1, key2], value2:[key1]} """
    values = list(set(dict_values_list(dic)))
    val_key_map = {}
    for val in values:
        val_key_map[val] = []
        for key,val_list in dic.items():
            if val in val_list:
                val_key_map[val].append(key)
    return val_key_map

def dict_values_list(d):
    """ Input: dictionary with objects or lists as values, d->[x]. Returns all values in a single list"""
    lists = d.values()
    all_list = []
    for l in lists:
        if is_list(l):
            all_list.extend(l)
        else:
            all_list.append(l)
    return all_list

def dict_update_nested(d, d_upd):
    """
    update nested dictionary d with partial update-dictionary d_upd.
    """
    for k,v in d_upd.items():
        if k in d:
            if isinstance(v, dict):
                dict_update_nested(d[k], v)
            else:
                d[k] = v
        else:
            d[k] = v

def dict_of_lists_to_dicts(dict_of_lists):
    lens = list(set([len(v) for v in dict_of_lists.values()]))
    if len(lens)!=1:
        raise ValueError('lists have different lengths: %s' % str(lens))
    n = lens[0]

    dicts = []
    for i in range(n):
        di = {}
        dicts.append(di)
        for k,vals in dict_of_lists.items():
            di[k] = vals[i]

    return dicts

def dict_add_values(d1, d2):
    """ mak a new dictionary with added values of the given dictionaries """
    common_keys = list_intersection(list(d1.keys()), list(d2.keys()))
    assert len(common_keys)==len(d1.keys()), 'dictionaries must have same keys %s, %s' % (str(d1.keys(), str(d2.keys())))
    d = {}
    for k,v in d1.items():
        d[k] = d1[k]+d2[k]
    return d

def arrays_mean(arrays):
    return np.mean( np.array(arrays), axis=0 )
def arrays_std(arrays):
    return np.std( np.array(arrays), axis=0 )

def unique_one(series, allowna=True):
    """ expects and returns the single unique value in the series, throws an error otherwise"""
    if is_list_or_tuple(series):
        sname = 'list'
    else:
        sname = series.name
    vals = list(set(series))
    vals_notna = [v for v in vals if str(v)!='nan']
    if not allowna and len(vals)!=len(vals_notna):
        raise ValueError(f'series {sname} contains nans: {vals}')

    if len(vals_notna)>1: #multiple values
        raise ValueError('%d>1 unique values %s in series %s!' % \
                         (len(vals), str(vals), sname))
    elif len(vals_notna)==1:
        return vals_notna[0] #single vbalue
    else:
        return vals[0] # nan

def dict_agg_type(dicts, number_unique=False, string_unique=False, rest_unique=False, arr1d_unique=False,
                  only_common_keys=False, ignore_keys=[]):
    if len(dicts)==0:
        return None
    elif len(dicts)==1:
        return dicts[0]
    n = len(dicts)
    keys = list(dicts[0].keys())
    keys = [k for k in keys if k not in ignore_keys]
    for d in dicts[1:]:
        dkeys = set(d.keys())
        if only_common_keys:
            keys = list_intersection(keys, dkeys)
        elif set(keys)!=dkeys:
            raise ValueError('aggregation not possible, different keys:', keys, dkeys)
    result = {}
    for key in keys:
        values = [d.get(key) for d in dicts if key in d]
        values = [v.item() if len(v.shape)==0 else v for v in values]
        is_num = all([isinstance(v, (int, float)) for v in values])
        if is_num:
            if number_unique:
                result[key] = unique_one(values)
            else:
                result[key] = np.mean(values)
                result[f"{key}_std"] = np.std(values)
        elif all([isinstance(v, np.ndarray) for v in values]):
            if arr1d_unique and len(values[0].shape)==1:
                result[key] = unique_one(values)
            else:
                result[key] = arrays_mean(values)
                result[f"{key}_std"] = arrays_std(values)
        elif all([isinstance(v, str) for v in values]):
            if string_unique:
                result[key] = unique_one(values)
            else:
                result[key]= ';'.join(values)
        else:
            if rest_unique:
                result[key] = unique_one(values)
            else:
                result[key] = [v for v in values if v is not None]
    return result

def _test_dict_lists_to_lists_dict():
    dict_lists = {'a':[0,1],'b':['b0','b2']}
    ld = dict_of_lists_to_dicts(dict_lists)
    print(ld)

class Timer(object):
    def __init__(self, name=None, start=True, verbose=True):
        self.name = name
        self.verbose = verbose
        self.running = False
        self.elapsed = 0
        if start:
            self.start()

    def start(self):
        self.time = timeit.default_timer()
        self.running = True

    def stop(self, restart=False):
        if not self.running:
            return 0

        self.running = False
        current_time = timeit.default_timer()
        elapsed = current_time - self.time
        self.elapsed += elapsed
        self._print_time(elapsed)
        if restart:
            self.start()
        return elapsed

    def print(self):
        return self._print_time(self.elapsed, suffix=' overall')

    def _print_time(self, elapsed, suffix=''):
        time_str = 'time: '+time_to_str(elapsed)
        if self.name is not None:
            time_str = f'{self.name}{suffix} {time_str}'
        print(time_str)
        return time_str



class RunEst(object):
    """ Estimates runtime given a number of tasks """
    def __init__(self, start=False, n_tasks=None, remember=3, print_fct=print, task_name='Task'):
        self.elapsed = collections.deque(maxlen=remember)
        self.count = 0
        self.running = False
        self.n_tasks = n_tasks
        self.print_fct = print_fct
        self.task_name = task_name
        if start: self.start()

    def start(self):
        self.start_time = time.time()
        self.running = True
        return self

    def stop(self, n=1, ret_str=True, print_remaining_string=False):
        try:
            self.elapsed_since_last = time.time()-self.start_time
            self.elapsed.append(self.elapsed_since_last)
            self.count += n
            self.running = False
            if print_remaining_string:
                self.print_fct(self.remaining_string())

            if ret_str:
                return self.elapsed_since_last_string()
            else:
                return self.elapsed_since_last
        except:
            self.print_fct('run est failed:')
            print(sys.exc_info())

    def tick(self, n=1, ret_str=True, print_remaining_string=True):
        self.stop(n=n, ret_str=ret_str, print_remaining_string=print_remaining_string)
        if self.count!=self.n_tasks:
            self.start()

    def remaining(self, n=None):
        if n is None:
            n = self.n_tasks - self.count
        time_per_task = np.mean(self.elapsed)
        remaining = int(n*time_per_task)
        return remaining

    def elapsed_since_last_string(self):
        return time_to_str(self.elapsed_since_last)

    def print_elapsed(self, prefix=''):
        stri = self.elapsed_since_last_string()
        if prefix is not None and len(prefix)>0:
            stri = prefix +' ' + stri
        print(stri)

    def _time_string(self, n=1):
        if self.count <= 0:
            return '0'

        est_time = self.remaining(n)

        return time_to_str(est_time)

    def remaining_string(self, n=None, prefix='ETA: '):
        try:
            if n is None:
                n = self.n_tasks - self.count
            time_string = self._time_string(n)
            return '%s %d/%d ' % (self.task_name, self.count, self.n_tasks)+prefix+time_string
        except Exception as ex:
            print('Error during remaining_time_string: %s' % str(ex))
            return ""

    def remaining_string_ext(self, n=1, ext='last'):
        time_string = 'eta='+self._time_string(n)
        if self.remaining(1) > 0:
            if ext=='avg':
                time_string += ', avg=%s' % (self._time_string(1))
            elif ext=='last':
                time_string += ', time=%s' % (time_to_str(self.elapsed_since_last))
            else: raise ValueError('unknown extra info type %s' % ext)
        return time_string


    def avg_speed(self):
        if len(self.elapsed) == 0:
            return -1
        return np.mean(self.elapsed)

    def avg_speed_string(self):
        if len(self.elapsed) == 0:
            return '-1'
        else:
            return time_to_str(self.avg_speed())

def call_ls(d):
    return subprocess.call(["ls",  "-ld", str(d)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def can_open_file(path, ls_parent=False):
    try:
        # os.system("ls -ld " + str(path))
        for i in range(5):
            subprocess.call(["ls",  "-ld", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        f = open(str(path), 'r')
        f.close()
        return True
    except IOError:
        if ls_parent:
            # os.system("ls " + str(Path(path).parent))
            subprocess.call(["ls", str(Path(path).parent)], stdout=subprocess.DEVNULL)
            return can_open_file(path, ls_parent=False)
        return False


def dict_to_dict_with_keys_as(d, fct):
    """ Returns a copy of the dictionary with the non-string keys (recursively) replaced with their string-versions"""
    ret_list = True
    if not is_list(d):
        ret_list = False
        d = [d]

    dl = []
    for di in d:
        new_dict = {}; dl.append(new_dict)
        for k, v in di.items():
            if isinstance(v, dict):
                v = dict_to_dict_with_keys_as(v, fct)
            new_dict[fct(k)] = v
    if ret_list: return dl
    return new_dict

def dict_add_key_prefx(d:dict, prefix):
    d_ = {}
    for k,v in d.items():
        d_[prefix+str(k)] = v
    return d_

def dict_nested_update(d, d_upd):
    """
    update nested dictionary d with partial update-dictionary d_upd.
    """
    for k,v in d_upd.items():
        if k in d:
            if isinstance(v, dict):
                dict_update_nested(d[k], v)
            else:
                d[k] = v
        else:
            d[k] = v

def dict_nested_get(d, key, default=None, separator='.'):
    keys = key.split(separator)
    for k in keys:
        if k not in d:
            return default
        d = d[k]
    return d

def dict_to_csv(info_dict, save_path, sep=','):
    df_info = pd.DataFrame.from_dict(info_dict.items())
    df_info.style.set_properties(**{'text-align': 'left'})
    df_info.to_csv(str(save_path), index=None, header=None, sep=sep)

def dict_from_csv(path, sep=','):
    df = pd.read_csv(str(path), index_col=None, header=None, sep=sep) #cols 0: name 1: value
    df = df.set_index(0)[1]
    d = df.to_dict()
    return d



class JsonNpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return super().default(obj)

def write_json_dict(data, path, convert_keys_to_strings=False):
    if convert_keys_to_strings:
        data = dict_to_dict_with_keys_as(data, fct=str)
    # data = dict_np_values_as_items(data)
    parent = Path(path).parent
    mkdir(parent)
    with open(str(path), 'w') as outfile:
        json.dump(data, outfile, indent=4, cls=JsonNpEncoder)

def read_json_dict(path, convert_keys_fct=None):
    with open(str(path), 'r') as f:
        data = json.load(f)
    if convert_keys_fct is not None:
        data = dict_to_dict_with_keys_as(data, convert_keys_fct)
    return data

def read_json(path, convert_keys_fct=None):
    return read_json_dict(path, convert_keys_fct)

def write_json(data, path, convert_keys_to_strings=False):
    return write_json_dict(data, path, convert_keys_to_strings=convert_keys_to_strings)

def dict_np_values_as_items(dic):
    ret_list = True
    if not is_list(dic):
        dic = [dic]
        ret_list = False

    d_list = []
    for di in dic:
        d = {}; d_list.append(d)
        for k,v in di.items():
            if 'numpy' in str(type(v)):
                v = v.item()
            d[k] = v
    if ret_list: return d_list
    else: return d #only one dictionary

def dict_get_existing_keys(d, keys):
    extracted = {k:d[k] for k in keys if k in d}
    return extracted

def dict_sorted_values_by_key(dict):
    return list(collections.OrderedDict(sorted(dict.items())).values())

def invert_dict(d):
    inverted = dict([[v, k] for k, v in d.items()])
    return inverted

def invert_dict_of_unique_lists(d):
    """ input: {key1:[val1, val2]} Returns: {val1: key1, val2: key1}"""
    inverted = {}
    for k,vals in d.items():
        for val in vals:
            if val in inverted:
                raise ValueError('%s occurs several times' % str(val))
            inverted[val] = k
    return inverted

def invert_dict_non_unique(d):
    "input: key->val, keys unique, vals not unique, returns val->[keys]"
    values = list(set(d.values()))
    value_key_map = defaultdict(list)
    for value in values:
        for key, v in d.items():
            if value==v:
                value_key_map[value].append(key)
    return value_key_map


#https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
class DefaultOrderedDict(collections.OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
                not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        collections.OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return collections.OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                               collections.OrderedDict.__repr__(self))


def copy_file(src, dst, dry_run=False):
    if src is None or dst is None:
        print('copying not possible: src or dst is None, src: %s, dst: %s' % (str(src), str(dst)))
        return
    if Path(dst).is_dir():
        dst = Path(dst)/Path(src).name
    if dry_run:
        print('not copying %s to %s' % (str(src), str(dst)))
        return None
    with open(str(src), 'rb') as fin:
        with open(str(dst), 'wb') as fout:
            shutil.copyfileobj(fin, fout, 128 * 1024)
    # return shutil.copyfile(str(src), str(dst))
    return dst

def hwc_to_chw(arr):
    if len(arr.shape)==3:
        return arr.transpose((2, 0, 1))
    elif len(arr.shape)==4:
        return arr.transpose((0, 3, 1, 2))
    else:
        raise ValueError('unknown shape', arr.shape)

def chw_to_hwc(arr):
    if len(arr.shape)==3:
        return arr.transpose((1, 2, 0))
    elif len(arr.shape)==4:
        return arr.transpose((0, 2, 3, 1))
    else:
        raise ValueError('unknown shape', arr.shape)



def nested_dict_flatten(d, prefix='', separator='.'):
    """ transforms nested dictionary to a falt representation,
    e.g {a:{x:1, y:{z:2}}} -> {a.x:1, a.y.z:2} """
    d_flat = {}
    if prefix is not None and len(prefix) > 0:
        prefix = prefix + separator
    else:
        prefix = ''

    for k,v in d.items():
        if isinstance(v, dict):
            d_f = nested_dict_flatten(v, prefix=prefix+str(k), separator=separator)
            d_flat.update(d_f)
        else:
            if prefix is not None and len(prefix)>0:
                k = prefix + str(k)
            d_flat[k] = v
    return d_flat

def dict_tree(depth, leaf):
    if depth>0:
        fct = partial(dict_tree, depth=-1, leaf=leaf)
    else:
        fct = leaf
    return collections.defaultdict(fct)

def time_to_str(est_time:float):
    """ time in seconds to string """
    if est_time == 0:
        return '0s'
    elif est_time < 1:
        if est_time > 1e-4:
            return '%.4fs' % est_time
        else:
            return '%.2fms' % (est_time*1000)

    minutes, seconds = divmod(est_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return '%dd%dh%02dm%02ds' % (days, hours, minutes, seconds)
    elif hours > 0:
        return '%dh%02dm%02ds' % (hours, minutes, seconds)
    elif minutes > 0:
        return '%02dm%02ds' % (minutes, seconds)
    elif seconds > 0:
        return '%ds' % seconds
    else:
        return ''


def center_crop_img(img, target_size):
    height, width, depth = img.shape
    if not is_list_or_tuple(target_size):
        target_size = (target_size, target_size)
    if height == target_size[0] and width == target_size[1]:
        return img
    diff_y = (height - target_size[0]) // 2
    diff_x = (width - target_size[1]) // 2
    return img[diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

def center_pad_img(arr, size, fill=0):
    """ expects the features to be in the third dimension """
    if not is_iterable(size):
        size = (size, size)
    x, y = arr.shape[:2]
    diffx = size[0]-x
    diffy = size[1]-y
    if diffx < 0 or diffy < 0:
        raise ValueError(f'center padding not possible, image {x,y} is bigger than padded size {size}')
    padx1 = diffx//2
    padx2 = diffx-padx1
    pady1 = diffy//2
    pady2 = diffy-pady1

    padding = ((padx1, padx2), (pady1, pady2), (0, 0))
    padded = np.pad(arr, padding, 'constant', constant_values=fill)
    return padded

def remap_arr(arr, remap:dict):
    """ remaps values in the array {old:new} """
    if remap is None or len(remap)==0:
        return
    ind_vals = []
    for oldval, newval in remap.items():
        ind_vals.append((np.where(arr == oldval),newval))

    for ind,val in ind_vals:
        arr[ind] = val

def print_pythonpath():
    print(os.environ['PYTHONPATH'])

def print_env_vars():
    print('ENV VARS:')
    stris = []
    for k, v in sorted(os.environ.items()):
        stris.append(k + ': ' + v)
    stri = ', '.join(stris)
    print(stri)

def print_env_info(verbose=True):
    print('PATH', os.environ['PATH'])
    print_pythonpath()
    print('CONDA_PREFIX', os.environ.get('CONDA_PREFIX',''))
    if verbose:
        print_env_vars()
        import numpy
        print('numpy version', numpy.__version__)


def _test_dict_nested_flatten():
    d = {'a':{'x':1, 'y':{23874:2}}}
    d_exp = {'a.x':1, 'a.y.23874':2}
    d_flat = nested_dict_flatten(d)
    print(d)
    print(d_flat)
    assert d_flat==d_exp

class timer(object):
    def __init__(self, description=None, verbose=True):
        self.description = description
        self.verbose = verbose
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, type, value, traceback):
        self.end = time.time()
        if self.description is None:
            self.description = ''
        else:
            self.description+=' '
        run_time = self.end-self.start
        self.time = run_time
        run_time = time_to_str(run_time)
        if self.verbose:
            print(f"{self.description}time: {run_time}")

def rows_cols_for_gridplot(images):
    if is_iterable(images):
        n_images = len(images)
    else:
        n_images = images
    ncols = int(np.ceil(np.sqrt(n_images)))
    nrows = int(np.floor(np.sqrt(n_images)))
    if ncols*nrows < n_images:
        nrows+=1
    return nrows, ncols

def plot_imagegrid(images, rows=None, cols=None, figsize=(10,10), show=True, titles=None, wspace=0, hspace=0,
                   save_path=None, **kwargs):
    if isinstance(images, np.ndarray):
        images = np.squeeze(images)

    if rows is None:
        cols = int(np.ceil(np.sqrt(len(images))))
        rows = int(np.floor(np.sqrt(len(images))))
    plot_images(images, rows, cols, figsize, show, titles, wspace, hspace,
                save_path=save_path, **kwargs)

def hide_axis(ax, show_axis=False):
    if show_axis:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
    else:
        ax.set_axis_off()

def plot_images(images, rows, cols, figsize=(10,10), show=True, titles=None,
                wspace=0, hspace=0, title_pad=0, title_bold=False, save_path=None, show_axis=False):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize,
                             gridspec_kw={'wspace': wspace, 'hspace': hspace})

    axes = list(axes.flat)
    for i,arr in enumerate(images):
        ax = axes[i]
        im = ax.imshow(np.squeeze(images[i]))
        hide_axis(ax, show_axis=show_axis)
        if titles is not None and titles is not False:
            options = {}
            if title_bold:
                options.update(dict(fontweight="bold"))
            if title_pad!=0:
                options.update(dict(pad=title_pad,  y=1.000001))
            ax.set_title(titles[i], fontsize=10, **options)
    for j in range(i+1,len(axes)):
        hide_axis(axes[j])

    # fig.subplots_adjust(right=0.85)
    # #[left, bottom, width, height
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def showgrid(images, *args, **kwargs):
    plot_imagegrid(images, *args, **kwargs)

def showim(arr, cb=False, save_path=None, showaxis=False, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    im = ax.imshow(arr)
    if not showaxis:
        ax.set_axis_off()#.axis('off')
    if cb:
        fig.colorbar(im, orientation='vertical')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight")
    plt.show()

def showims(*arrs, figsize=(10,20), rows=None, cols=None, titles=None, sharexy=False, sharex=False, sharey=False,
            noaxes=False, wspace=None, hspace=None, **kwargs):
    if rows is None:
        rows = 1
    if cols is None:
        cols = int(np.ceil(len(arrs)/rows))

    if sharexy: sharex = sharey = True
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=sharex, sharey=sharey)
    if cols==1:
        axes = [axes]
    else:
        axes = list(axes.flat)
    for i, arr in enumerate(arrs):
        if titles is not None: axes[i].set_title(titles[i])
        axes[i].imshow(arr, **kwargs)
        if noaxes:
            axes[i].axis('off')
    plt.tight_layout()
    if wspace is not None:
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()

def showx(*arr, figsize=(10,10), hm=False, title=[]): #for debugging
    """ shows the arrays in a single row """
    n_cols = len(arr)
    fig, axes = plt.subplots(nrows=1, ncols=len(arr), figsize=figsize)
    if n_cols==1:
        axes = [axes]
    else:
        axes = list(axes.flat)
    if title is not None and is_string(title):
        titles = [title]
    vmin = None; vmax = None
    if hm:
        vmin=0; vmax = 1
    for i, ax in enumerate(axes):
        if arr[i].dtype==np.uint8:
            vmax = 255
        im = ax.imshow(arr[i].squeeze(), vmin=vmin, vmax=vmax)
        if title is not None and len(title)>i:
            ax.set_title(title[i])
    if hm:
        # # cbar = fig.colorbar(im, ax=ax, shrink=0.5)
        # cbar = fig.colorbar(im, ax=ax)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)

    # plt.show(block=True)
    print('plt.show')
    plt.show()

def bar_autolabel_horizontal(ax, rects=None, formatter='%.3f', xoffset=0.1, inside=False, bold=True):
    if rects is None:
        rects = ax.patches
    rect_widths = sorted([rect.get_width() for rect in rects])
    xoffset = rect_widths[len(rect_widths)//2]*xoffset
    for rect in rects:
        width = rect.get_width()
        d = dict(y=rect.get_y() + 0.5 * rect.get_height(), s=formatter % width, ha='center', va='center')
        if bold:
            d['fontweight'] = 'bold'
        if inside:
            d.update(dict(x=rect.get_width() - xoffset, color='white'))
        else:
            d.update(dict(x=rect.get_width() + xoffset))
            # ax.text(rect.get_width()+xoffset, rect.get_y()+0.5*rect.get_height(),
            #      formatter % width, ha='center', va='center')
        ax.text(**d)

def _is_dir_check(f, fast_and_dirty=False):
    """ if fast_and_dirty checks only the extension"""
    if fast_and_dirty:
        return f.suffix is None or len(f.suffix)==0
    else:
        return f.is_dir()

def pad_img_to_size(img_arr, size, fill=0):
    pad_width = max(size[1] - img_arr.shape[1], 0)
    pad_height = max(size[0] - img_arr.shape[0], 0)
    if pad_width>0 or pad_height > 0:
        img_arr = np.pad(img_arr, ((math.floor(pad_height / 2), math.ceil(pad_height / 2)),
                                   (math.floor(pad_width / 2), math.ceil(pad_width / 2)),
                                   (0, 0)), mode='constant', constant_values=fill)
    return img_arr

class CaptureOut(list):
    """ captures sys out prints as list """
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def capture_out_example():
    print('hello!')
    print('hello again!')


#old
def pathes_in(d, recursive=True, recursive_depth=10, starting=None, ending=None, containing=[], containing_or=[], not_containing=[],
              files_only=True, dirs_only=False, sort=False, sort_by_size=False, as_strings=False, print_progress=False,
              diry_dir_check=False):
    """ returns the contents of the given directory"""
    if files_only and dirs_only:
        raise ValueError('either files_only or dirs_only, not both')
    pathes = []
    if is_string(d):
        d = Path(d)
    if not_containing is None:
        not_containing=[]
    elif is_string(not_containing):
        not_containing = [not_containing]
    if ending is None:
        ending = []
    elif is_string(ending):
        ending = [ending]
    if containing is None:
        containing = []
    if containing_or is None:
        containing_or = []
    elif is_string(containing):
        containing = [containing]
    elif is_string(containing_or):
        containing_or = [containing_or]

    if sort_by_size:
        def _get_size(entry):
            return entry.stat().st_size

        all_files = sorted(d.iterdir(), key=_get_size)
    else:
        all_files = [f for f in d.iterdir()]

    if print_progress:
        from tqdm import tqdm
        pbar = tqdm(total=len(all_files))

    for f in all_files:
        if recursive and recursive_depth>0 and f.is_dir():
            files_in_dir = pathes_in(f, recursive=True, starting=starting, recursive_depth=recursive_depth-1,
                                     containing=containing, not_containing=not_containing, ending=ending)
            pathes.extend(files_in_dir)
        selected=True
        if files_only and _is_dir_check(f, diry_dir_check):
            selected = False
        if dirs_only and f.is_file():
            selected = False

        if starting is not None and not f.name.startswith(starting):
            selected = False
        for co in containing:
            if not co.lower() in str(f).lower():
                selected = False
                break
        if len(containing_or)>0:
            found = False
            for co in containing_or:
                if co.lower() in str(f).lower():
                    found = True
                    break
            if not found:
                selected = False

        for nc in not_containing:
            if nc.lower() in str(f).lower():
                selected = False
                break

        if len(ending) > 0:
            found = False
            for end in ending:
                if f.name.endswith(end):
                    found = True
                    break
            if not found:
                selected = False

        if selected:
            pathes.append(f)
        if print_progress:
            pbar.update(1)
    if print_progress:
        pbar.close()
    if sort:
        pathes = sorted(pathes)
    if as_strings:
        pathes = pathes_to_string(pathes)
    return pathes

#old
def dirs_in(d, recursive=False, **kwargs):
    return pathes_in(d, files_only=False, dirs_only=True, recursive=recursive, **kwargs)


def string_in_list(s, lis):
    for obj in lis:
        if s == str(obj):
            return True
    return False

def string_insert (src, pos, insert):
    return src[:pos]+insert+src[pos:]

def pathes_to_string(pathes, name_only=False, stem_only=False, sort=False):
    if name_only and stem_only:
        raise ValueError('either name_only or stem_only, not both')
    strings = []
    for path in pathes:
        if name_only:
            strings.append(path.name)
        elif stem_only:
            strings.append(path.stem)
        else:
            strings.append(str(path))
    if sort:
        strings = sorted(strings)
    return strings

def df_row_to_dict(df, ind=0):
    return df.to_dict('records')[ind]

def file_size(path, mb=True):
    return get_file_size(path, mb)

def get_file_size(path, mb=False):
    """ returns the filesize in bytes or megabytes if mb=True"""
    fs = float(round(os.path.getsize(str(path))))
    if mb:
        fs = fs/1000000
    return fs

def get_dir_size(directory, mb=False):
    """Returns the `directory` size in bytes."""
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                total += get_dir_size(entry.path)
    except NotADirectoryError:
        total = os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    if mb:
        total = total / 1000000
    return total

def save_arrays(save_arrays_path, *arrays, **named_arrays):
    save_arrays_path = str(save_arrays_path)
    if arrays is None and named_arrays is None:
        print('Nothing to save')
    try:
        np.savez_compressed(save_arrays_path, *arrays, **named_arrays)
    except:
        print("Error: %s" % str(sys.exc_info()))
        print('Waiting one minute before trying again..')
        time.sleep(3)
        np.savez_compressed(save_arrays_path, *arrays, **named_arrays)

    # print(save_arrays_path + ' saved')

def fct_from_str(fct_str):
    """ fct_str: module.fct """
    parts = fct_str.split('.')
    if len(parts)<2: raise ValueError('expects functions in form of module.fct')
    module_str = '.'.join(parts[:-1])
    modul = importlib.import_module(module_str)
    fct = getattr(modul, parts[-1])
    return fct

def multiproc_wrapper(kwargs):
    """ expects the function to be conained in the args under '_function' """
    fct = kwargs.pop('_function')
    if is_string(fct):
        fct = fct_from_str(fct)
    return fct(**kwargs)

def multiproc_pool(fct, kwargs, n_workers_min=2):
    if not is_string(fct):
        modul = fct.__module__
        fct = modul+'.'+fct.__name__
    n_cpus = max(n_workers_min,count_docker_cpus())
    pool = multiprocessing.Pool(processes=n_cpus)
    n_tasks = len(kwargs)
    for kwarg in kwargs:
        kwarg['_function'] = fct
    print('Starting pool with function %s, %d workers on %d args' % (fct, n_cpus, len(kwargs)), flush=True)

    # results = pool.map(multiproc_wrapper, kwargs)
    # if join:
    #     pool.close() #parallelizable part of your main program is finished.
    #     pool.join() #wait for the worker processes to terminate
    #     return results
    # else:
    #     return pool

    run_est = RunEst(n_tasks=len(kwargs), start=True)
    results = []
    for result in pool.imap_unordered(multiproc_wrapper, kwargs):
        results.append(result)
        run_est.stop(n=1, print_remaining_string=True)
        run_est.start()
    return results

from concurrent.futures import ProcessPoolExecutor
def multiproc_pool2(fct, kwargs, cpus=2, verbose=True):
    if cpus<=1:
        return sequential_pool(fct, kwargs)

    start_time = time.time()
    if not is_string(fct):
        modul = fct.__module__
    fct = modul+'.'+fct.__name__
    for kwarg in kwargs:
        kwarg['_function'] = fct
    print('Starting ProcessPoolExecutor with function %s, %d workers on %d args' % (fct, cpus, len(kwargs)), flush=True)

    if verbose: run_est = RunEst(n_tasks=len(kwargs), start=True)
    results = []
    with ProcessPoolExecutor(max_workers=cpus) as pool:
        results_iter = pool.map(multiproc_wrapper, kwargs)
        for result in results_iter:
            results.append(result)
            if verbose: run_est.tick(n=1, print_remaining_string=True)
    run_time = time.time() - start_time
    print('Time: {delta}'.format(delta=datetime.timedelta(seconds=run_time)))

    return results

def sequential_pool(fct, kwargs, **ignore_kwargs):
    results = []
    run_est = RunEst(n_tasks=len(kwargs), start=True)
    for arg in tqdm(kwargs):
        results.append(fct(**arg))
        run_est.tick()
    return results

def print_mp(*args, flush=True, **kwargs):
    """ prints with the pid as prefix if the process is not the main process """
    is_main = multiprocessing.current_process().name == 'MainProcess'

    now = datetime.datetime.now()
    prefix = '%02d:%02d' % (now.hour, now.minute)
    # prefix = '%02d:%02d:%02d' % (now.hour, now.minute, now.second)

    if not is_main:
        pid = multiprocessing.current_process().ident
        builtins.print(prefix, str(pid) + ':', *args, flush=flush, **kwargs)
    else:
        builtins.print(prefix, *args, flush=flush, **kwargs)

def print_mp_err(*args, flush=True, **kwargs):
    return print_mp(*args, flush=flush, file=sys.stderr, **kwargs)

def write_yaml_dict(out_path, data, overwrite=False):
    if Path(out_path).exists() and not overwrite:
        raise ValueError('not overwriting %s' % out_path)
    ensure_dir_exists(Path(out_path).parent)
    with open(str(out_path), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    print('%s' % out_path)


def read_yaml_dict(in_path):
    with open(in_path) as file:
        dict = yaml.load(file, Loader=yaml.SafeLoader)
    return dict

def read_lines(path, ignore_empty=True, sort=False):
    with open(str(path)) as f:
        lines = f.read().splitlines()
    if ignore_empty:
        non_empty_lines = [l for l in lines if len(l)>0]
        lines = non_empty_lines
    if sort:
        lines = sorted(lines)
    return lines

def write_lines(path, lines, overwrite=True):
    if is_iterable(path) and not is_iterable(lines):
        #switched
        lines_ = lines
        lines = path
        path = lines_
    if not overwrite and path_exists(path):
        print('not overwriting %s' % str(path))
        return
    with open(str(path), 'w') as file:
        for line in lines:
            file.write(line+'\n')

class ParamInfo(object):
    """ class to save the parameters as a yaml and ensure that they not conflict when restarting
    the program """

    def __init__(self, out_dir, filename='params.yaml', overwrite=False):
        self.out_path = Path(out_dir) / (filename + '' if filename.endswith('.yaml') else '.yaml')
        self.overwrite = overwrite

    def save(self, **kwargs):
        result = self._conflict_check(**kwargs)
        if str(result)!='same':
            write_yaml_dict(self.out_path, kwargs, overwrite=True)

    def _conflict_check(self, **kwargs):
        """ returns 'same' if file already exists and is same as kwargs, otherwise 'different'
        or throws an error if overwrite=False. If file is not there, return 'nofile' """
        if self.out_path.exists():
            other_args = read_yaml_dict(self.out_path)
            if kwargs != other_args:
                msg = f'warning: difference in args, old: {other_args}, new: {kwargs}'
                self.out_path.absolute()
                if self.overwrite:
                    print(msg)
                    return 'different'
                else:
                    raise ValueError(msg +  '. delete %s if you want to process the slides anyway' % self.out_path)
            else:
                return 'same'
        else:
            return 'nofile'

def run_cmd(cmd, dry_run=False):
    if dry_run:
        print('not running %s' % str(cmd))
        return
    lines = subprocess.getoutput(cmd).splitlines()
    return lines

def run_cmd_live(cmd, dry_run=False):
    """ runs the command with live output """
    print('cmd: %s' % cmd)
    if dry_run: return
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.strip())
    p.stdout.close()
    p.wait()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def create_tensorboard_start_text(tbx_dir):
    lines = ['#!/bin/bash',
             '#tensorboard --logdir=%s' % str(tbx_dir),
             'gnome-terminal -x bash -c "tensorboard --logdir=%s; ; exec bash"' % str(tbx_dir)
             ]

    # lines = ['tensorboard --logdir=%s' % str(Path(tbx_dir).absolute())]
    out_path = Path(tbx_dir)/'tensorboard.sh'
    ensure_dir_exists(tbx_dir)
    write_lines(lines, out_path)

def ignore_warnings():
    warnings.filterwarnings('ignore', '.*output shape of zoom.*')
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)



def string_empty(text):
    return text is None or (is_string(text) and is_empty(text)) or (not is_string(text) and np.isnan(text))


def take_closest_smaller_number(l, number):
    smaller = [x for x in l if x <= number]
    if not smaller:
        raise ValueError(f'no value in {l} is smaller or equal to {number}')
    return max(smaller)

if __name__ == '__main__':
    pass
    # _test_dict_nested_flatten()
    # run_cmd_live('whoami')
    # _test_dict_lists_to_lists_dict()
    print(most_frequent(['a','x','x', 'x','b','b']))
    print(most_frequent([4, 2, 2, 2, 0, 0]))
