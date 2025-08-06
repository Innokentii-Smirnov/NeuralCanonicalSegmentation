from . import get_file_name as get_name
from collections import OrderedDict

def read_dict(file_name: str) -> dict[str, str]:
    d = dict()
    with open(get_name(file_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            try:
                arr = list(filter(lambda x: x != '',
                           line.removesuffix('\n').split('\t')))
                key, value = arr
            except ValueError:
                raise ValueError(str(arr))
            d[key] = value
    return d

def read_list(file_name: str) -> list[str]:
    l = list()
    with open(get_name(file_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            item = line.removesuffix('\n')
            l.append(item)
    return l

def read_text(file_name: str) -> str:
    with open(get_name(file_name), 'r', encoding='utf-8') as fin:
        return fin.read()

def read_multidict(file_name: str) -> dict[str, list[str]]:
    data = map(lambda x: x.split('\t'), read_list(file_name))
    result = dict[str, list[str]]()
    for key, value in data:
        if key not in result:
            result[key] = list[str]()
        result[key].append(value)
    return result

def read_dict_of_list(file_name: str) -> dict[str, list[str]]:
    d = OrderedDict[str, list[str]]()
    with open(get_name(file_name), 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.startswith('\t'):
                key = line.removesuffix('\n')
                d[key] = list[str]()
            else:
                d[key].append(line.removeprefix('\t').removesuffix('\n'))
    return d
