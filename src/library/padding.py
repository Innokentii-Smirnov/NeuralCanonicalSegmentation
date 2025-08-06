from typing import TypeVar
T = TypeVar('T')

def pad(symbol: T, length: int, seq: list[T]) -> list[T]:
    return seq + [symbol] * (length - len(seq))
