from typing import Sequence

def pad(string: str, length: int):
    return string + ' ' * (length - len(string))

def align_column(sequence: Sequence[str]):
    maxlen = max(map(len, sequence))
    new = [pad(elem, maxlen) for elem in sequence]
    return new

def align(iterables: list[list[str]]):
    new = list[list[str]]()
    for _ in range(len(iterables)):
        new.append(list[str]())
    for col in range(len(iterables[0])):
        column = align_column([iterable[col] for iterable in iterables])
        for row, elem in enumerate(column):
            new[row].append(elem)
    return new

def align_inplace(iterable: list[str]):
    for i in range(len(iterable)):
        if iterable[i] is None:
            iterable[i] = ''
    maxlen = max(map(len, iterable))
    for i in range(len(iterable)):
        iterable[i] = pad(iterable[i], maxlen)
    return iterable
