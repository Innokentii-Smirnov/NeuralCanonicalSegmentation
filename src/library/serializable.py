from library.read import read_text
from library.write import write_text
from typing import TypeVar, Sequence
from functools import partial

def to_string(x):
    if isinstance(x, Serializable):
        return x.to_string()
    elif isinstance(x, str):
        return x
    else:
        return repr(x)

class Serializable:

    element_func = lambda x: x if x is x != 'None' else None
    sep = '%'

    def get_elements(self) -> Sequence:
        raise NotImplementedError
    
    def __tuple__(self) -> tuple:
        return tuple(self.get_elements())
    
    def __eq__(self, other) -> bool:
        return self.__tuple__() == other.__tuple__()

    def __lt__(self, other) -> bool:
        return self.__tuple__().__lt__(other.__tuple__())

    def __gt__(self, other) -> bool:
        return self.__tuple__().__gt__(other.__tuple__())

    def __hash__(self):
        return self.__tuple__().__hash__()

    @classmethod
    def from_strings(cls, *strings):
        raise NotImplementedError

    def to_string(self) -> str:
        return self.sep.join(map(to_string, self.get_elements()))
    
    def __repr__(self) -> str:
        return self.to_string()

    @classmethod
    def from_string(cls, string: str):
        return cls.from_strings(*list(map(cls.element_func, string.split(cls.sep))))

    def save(self, filename: str):
        write_text(self.to_string(), filename)

    @classmethod
    def load(cls, filename: str):
        return cls.from_string(read_text(filename))
    
T = TypeVar('T')

class SerializableList(Serializable, list[T]):
    
    @classmethod
    def get_element(cls, string: str) -> T:
        raise NotImplementedError

    def get_elements(self) -> Sequence[T]:
        return self

    @classmethod
    def from_string(cls, string: str):
        return cls(map(cls.get_element, string.split(cls.sep))) if string != '' else cls()
    
TKey = TypeVar('TKey')
TValue = TypeVar('TValue')

class SerializableDict(Serializable, dict[TKey, TValue]):
    separator = ' \u2192 '
    sep = ' \u222a '

    @classmethod
    def get_key(cls, string: str) -> TKey:
        raise NotImplementedError
    
    @classmethod
    def get_value(cls, string: str) -> TValue:
        raise NotImplementedError
    
    def get_elements(self) -> Sequence[str]:
        return [self.separator.join([to_string(key), to_string(value)]) for key, value in self.items()]

    @classmethod
    def from_string(cls, string: str):
        d = cls()
        if string != '':
            for elem in string.split(cls.sep):
                str_key, str_value = elem.split(cls.separator, 1)
                key = cls.get_key(str_key)
                value = cls.get_value(str_value)
                d[key] = value
        return d

class StringList(SerializableList):

    get_element = str

class StringStringDict(SerializableDict):

    get_key = str
    get_value = str

        
