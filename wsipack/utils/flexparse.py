import argparse, sys, yaml
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch
from wsipack.utils.parse_utils import parse_string
from wsipack.utils.cool_utils import dict_update_nested

def read_yaml(in_path):
    with open(in_path) as file:
        dict = yaml.load(file, Loader=yaml.SafeLoader)
    return dict

def parse_nested_dict(key, val):
    kparts = key.split('.')
    if len(kparts)==1:
        return {key:parse_string(val)}
    else:
        return {kparts[0]:parse_nested_dict('.'.join(kparts[1:]), val)}

class FlexArgumentParser(argparse.ArgumentParser):
    """ Goal: Flexibility, using the dictionaries that can be forwarded to the python classes directly as **kwargs.
    - nested dictionaries via '.'. , e.g. '--dict1.dict2.key=val'
    - automatic parsing of strings, ints, floats, bools and lists of these types, e.g.
      '--key=[12,1]'
    - support and recognition of yaml-configuration files, e.g.
      '--aconfig=/path/name.yaml' -> results in a (nested) dictionary 'aconfig' containing
       the yamls content
    - overwriting configuration entries, e.g. '--aconfig.somekey="new_value"'
    #Usage by others requires having an example - this is a feature, not a bug
    """
    def __init__(self, *args, parse_yaml=False, **kwargs):
        self.parse_yaml = parse_yaml
        super().__init__(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        known, unknown = self.parse_known_args(args=args, namespace=namespace)
        #add the unknown args as strings
        if unknown is not None:
            for arg in unknown:
                if arg.startswith(("-", "--")):
                    arg_name = arg.split('=')[0]
                    kwargs = {}
                    if '=' in arg:
                        kwargs['type']=str
                    else:
                        kwargs['action'] = 'store_true'
                    # you can pass any arguments to add_argument
                    self.add_argument(arg_name,**kwargs)
        #now parse all
        args = super().parse_args(args=args, namespace=namespace)
        args = vars(args)

        result = {}
        for k,v in args.items():
            if str(v).endswith('.yaml') and self.parse_yaml:
                if not Path(v).exists(): raise ValueError('configuration %s not found' % str(v))
                values = read_yaml(v)
                result[k] = values
        for key in result:
            del args[key]

        for k,v in args.items():
            try:
                parsed_dict = parse_nested_dict(k, v)
                dict_update_nested(result, parsed_dict)
            except:
                print('failed parsing:',k,v)
                raise
        return result


def _argparse_exp():
    parser = argparse.ArgumentParser()
    args, other = parser.parse_known_args(['--float1', '2.3', 'bool','True','--net_conf.depth', '2', '--net_conf.str', 'string'])
    print('args:',args)
    print('other args:', other)

def _flexparse_exp():
    parser = FlexArgumentParser()
    parser.add_argument('--net_conf.str', type=str)
    args = parser.parse_args(['--float1', '2.3', '--bool','True','--net_conf.depth', '2', '--net_conf.str', 'string'])
    print('args:',args)

def _flexparse_exp2():
    parser = FlexArgumentParser()
    # parser.add_argument('--net_conf.depth', type=str)
    args = parser.parse_args()
    print('args:',args)

if __name__ == '__main__':
    # _argparse_exp()
    # _flexparse_exp()

    # testargs = ['flexparse', '--float1', '2.3', '--bool','True','--net_config.depth=2', '--net_config.str', 'string', '--alist', '[10,20]']
    testargs = ['flexparse', '--alist', '[10,20]', '--wd', '5e-4']
    with patch.object(sys, 'argv', testargs):
        _flexparse_exp2()