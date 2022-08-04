import json, os
from json.decoder import JSONDecodeError


def _printable_path(pth):
    return os.path.relpath(pth,os.path.dirname(__file__)+'/..')

def dumpjson(dic, fname):
    dic = {int(key):value for key, value in dic.items()}
    with open(fname, 'w') as f:
        f.write(json.dumps(dic, indent=2, sort_keys=True))
    return fname

def dump_unsorted_json(dic, fname):
    with open(fname, 'w') as f:
        f.write(json.dumps(dic, indent=2))
    return fname

def loadjson(fname, strict = False):
    try :
        file = json.load(open(fname, 'r'))
        print(f'   -> Loaded {_printable_path(fname)}')
    except (FileNotFoundError, JSONDecodeError) as e:
        if strict:
            raise e
        file = {}
        print(f'  /!\ { _printable_path(fname)} not found -> returning empty dict')
    return file