import os
from os import path
extensions = ['.txt', '.xml', '.lex', '.csv', '.lfg']

default_dir = None

def get_file_name(file_name: str, use_default_dir = False, add_extension: bool = False) -> str:
    if use_default_dir and default_dir is not None and not file_name.startswith('/'):
        file_name = path.join(default_dir, file_name)
    if add_extension and '.' not in file_name:
        file_name += '.txt'
    return file_name
