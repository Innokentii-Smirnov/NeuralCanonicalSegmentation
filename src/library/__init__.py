import os
from os import path
extensions = ['.txt', '.xml', '.lex', '.csv', '.lfg']

default_dir = None

def get_file_name(file_name: str, use_default_dir = False) -> str:
    if use_default_dir and default_dir is not None and not file_name.startswith('/'):
        file_name = path.join(default_dir, file_name)
    if '.' not in file_name:
        file_name += '.txt'
    return file_name
