from wsipack.utils.cool_utils import is_string


def is_string_int(value):
    try:
        int(value)
        return True
    except:
        return False


def is_string_float(string):
    """ checks if the string is a float """
    try:
        float(string)
        return True
    except:
        return False

def has_numbers(stri):
    return any(char.isdigit() for char in stri)

def is_string_boolean(string):
    """ checks if the string is 'true' or 'false' case independent """
    low_string = string.lower()
    if low_string == 'true' or low_string == 'false':
        return True
    return False

def parse_string(param_value, list_separator=','):
    """ parses an int, float or string or lists or dictionaries of those types.
    alternative would be eval (unsafe) or something similar, but this should do.
    lists are either in the form of [i1,i2,...] or directly i1,2
    """
    if not isinstance(param_value, str):
        return param_value
    if len(param_value)==0:
        raise ValueError('empty string not allowed as parameter')

    if is_string_int(param_value):
        parsed_value = int(param_value)
    elif is_string_float(param_value):
        parsed_value = float(param_value)
    elif is_string_boolean(param_value):
        parsed_value = bool(param_value)
    elif (len(param_value)>2 and param_value[0]=='[' and param_value[-1]==']'):
        parts = param_value[1:-1].split(',')
        parsed_value = [parse_string(part.strip()) for part in parts]
    elif (param_value[0]=='{' and param_value[-1]=='}'):
        parts = param_value[1:-1].split(',')
        parsed_value = {}
        for part in parts:
            k,v = part.split(':')
            parsed_value[parse_string(k.strip())] = parse_string(v.strip())
    elif list_separator is not None and list_separator in str(param_value):
        parts = param_value.split(list_separator)
        parsed_value = [parse_string(part.strip()) for part in parts]
    else:
        parsed_value = param_value
    return parsed_value

def call_fct(name, *args, **kwargs):
    if is_string(name):
        parts = name.split('.')
        if len(parts)==1:
            func = globals()[name]
        else:
            module_name = '.'.join(parts[:-1])
            #correspoinds to from foo import bar
            module = __import__ (module_name, fromlist=['doesnt matter'])
            func = getattr(module, parts[-1])
    else:
        func = name
    return func(*args, **kwargs)

def _parse_test():
    assert 4==parse_string('4')
    assert 0.5==parse_string('0.5')
    assert True==parse_string('true')
    assert isinstance(parse_string('string'), str)

def _test_call_fct():
    res = call_fct('wutils.py_utils.random_string', length=5)
    assert len(res)==5

if __name__ == '__main__':
    # _parse_test()
    _test_call_fct()