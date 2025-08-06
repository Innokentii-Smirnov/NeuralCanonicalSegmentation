from typing import Sequence
from unicodedata import combining

LINE = 73

def string_length(string: str) -> int:
    length = 0
    for char in string:
        if combining(char) == 0:
            length += 1
    return length

def pad(string: str, target: int):
    return string + (target - string_length(string)) * " "

def sep(k: int):
    return '\n' + ' \n' * k

def subarr_to_str(arrays, start, end) -> str:
    return '\n'.join('  '.join(array[start:end]) for array in arrays)

def align(arrays: Sequence[list[str]]) -> str:
    copy = list[list[str]]()
    for i in range(len(arrays)):
        l = list[str]()
        copy.append(l)
    for j in range(len(arrays[0])):
        maxlength = max(string_length(array[j]) for array in arrays)
        for i in range(len(arrays)):
            copy[i].append(pad(arrays[i][j], maxlength))
    parts = []
    start = 0
    length = -2
    for w in range(len(copy[0])):
        length += len(copy[0][w]) + 2
        if length > LINE:
            if w > 0:
                part = subarr_to_str(copy, start, w)
                parts.append(part)
            start = w
            length = len(copy[0][w])
    parts.append(subarr_to_str(copy, start, len(copy[0])))
    return sep(1).join(parts)
