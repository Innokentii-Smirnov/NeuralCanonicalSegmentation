from typing import Callable, TypeVar
from functools import reduce
from itertools import chain
from collections.abc import Iterable

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
T = TypeVar('T')

def group_by(func: Callable[[TValue], TKey], iterable: Iterable[TValue]) -> dict[TKey, list[TValue]]:
    d = dict[TKey, list[TValue]]()
    for item in iterable:
        key = func(item)
        if key not in d:
            d[key] = list[TValue]()
        d[key].append(item)
    return d

def group_by_many(func: Callable[[TValue], Iterable[TKey]], iterable: Iterable[TValue]) -> dict[TKey, list[TValue]]:
    d = dict[TKey, list[TValue]]()
    for item in iterable:
        keys = func(item)
        for key in keys:
            if key not in d:
                d[key] = list[TValue]()
            d[key].append(item)
    return d

def modify_values_key(func, d1: dict[object, list[object]]):
    return {key: func(key, values) for key, values in d1.items()}

def modify_values(func, d1: dict[object, list[object]]):
    return {key: func(values) for key, values in d1.items()}

def modify_keys(func: Callable[[str], str],
                d: dict[str, set[str]]) -> dict[str, set[str]]:
    new_d = dict[str, set[str]]()
    for key, values in d.items():
        new_key = func(key)
        if new_key not in new_d:
            new_d[new_key] = values
        else:
            new_d[new_key] = new_d[new_key] | values
    return new_d

def count(func: Callable[[T], bool], iterable: Iterable[T]) -> int:
    return sum(1 for elem in iterable if func(elem))

def part(func: Callable[[T], bool], iterable: Iterable[T]) -> float:
    return count(func, iterable) / len(iterable)

def composition(func1: Callable, func2: Callable):
    return lambda x: func2(func1(x))

def compone(*funcs: Callable[[object], object]): 
    return reduce(composition, funcs)

def pi(i: int):
    return lambda iterable: list(
        map(lambda x: x[i],
        iterable)
    )

def chain_seq(sequences: Iterable[Iterable[T]]) -> Iterable[T]:
    return chain.from_iterable(sequences)

def find(condition: Callable[[object], bool], iterable: Iterable):
    for elem in iterable:
        if condition(elem):
            return elem
    return None

def item(iterable: Iterable[T]) -> T:
    sequence = list(iterable)
    assert len(sequence) == 1
    return sequence[0]
