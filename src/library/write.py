from . import get_file_name as get_name

def write_dict(d: dict, file_name: str, mode: str = 'w'):
    with open(get_name(file_name), mode, encoding='utf-8') as fout:
        for key, value in d:
            fout.write(str(key) + '\t' + str(value) + '\n')

def write_list(l: list, file_name: str, mode: str = 'w'):
    with open(get_name(file_name), mode, encoding='utf-8') as fout:
        for item in l:
            fout.write(str(item) + '\n')

def write_text(text: str, file_name: str, mode: str = 'w'):
    with open(get_name(file_name), mode, encoding='utf-8') as fout:
        fout.write(text)
