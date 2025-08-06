from collections import defaultdict
from collections.abc import Iterable
from typing import Optional
from library.read import read_list
from library.write import write_list

class Vocabulary:
    #sep = ';\n'
    extension = '.txt'
    #def get_elements(self) -> tuple[int, int, int, list[str]]:
    #    return self.add_begin, self.add_end, self.min_count, self.symbols_

    #@classmethod
    #def from_strings(cls, add_begin, add_end, min_count, symbols_):
    #    return cls(bool(add_begin), bool(add_end), int(min_count), symbols_.split())

    def __init__(self, add_begin: bool, add_end: bool,
                 min_count: int = 3, symbols: Optional[list[str]] = None):
        self.add_begin = add_begin
        self.add_end = add_end
        self.min_count = min_count
        self.symbols = symbols
    
    @property
    def symbols(self):
        return self.symbols_
    
    @symbols.setter
    def symbols(self, symbols):
        self.symbols_ = symbols
        if self.symbols_ is not None:
            self.symbol_codes_ = {letter: index for index, letter in enumerate(self.symbols_)}

    def fit(self, data: Iterable[Iterable[str]]):
        # специальные символы
        self.symbols_ = ["<PAD>", "<UNK>", "<BEGIN>", "<END>"]
        symbol_counts = defaultdict(int)
        for sequence in data:
            for elem in set(sequence):
                symbol_counts[elem] += 1
        self.symbols_ += sorted(letter for letter, count in symbol_counts.items() if count >= self.min_count)
        self.symbol_codes_ = {letter: index for index, letter in enumerate(self.symbols_)}
        return self

    @property
    def unk(self):
        return self.symbol_codes_["<UNK>"]

    @property
    def begin(self):
        return self.symbol_codes_["<BEGIN>"]

    @property
    def end(self):
        return self.symbol_codes_["<END>"]

    @property
    def pad(self):
        return self.symbol_codes_["<PAD>"]
    
    def vectorize_element(self, element: list[str]) -> list[int]:
        indexes = [self.symbol_codes_.get(symbol, self.unk) for symbol in element]
        if self.add_begin:
            indexes = [self.begin] + indexes
        if self.add_end:
            indexes = indexes + [self.end]
        return indexes
    
    def make_mask(self, item):
        answer = [True for _ in item]
        if self.add_begin:
            answer = [False] + answer
        if self.add_end:
            answer.append(False)
        return answer
    
    def __len__(self) -> int:
        return len(self.symbols_)
    
    def save(self, filename: str):
        write_list(self.symbols_, filename)
    
    def load(self, filename: str):
        self.symbols = read_list(filename)
    
    def __getitem__(self, key: str):
        return self.symbol_codes_[key]
    
    def __contains__(self, key: str):
        return key in self.symbol_codes_

class SequenceVocabulary(Vocabulary):

    #def get_elements(self) -> tuple[bool, bool, bool, bool, int, StringList]:
    #    return self.add_begin, self.add_end, self.add_begin_high, self.add_end_high, self.min_count, self.symbols_

    #@classmethod
    #def from_strings(cls, add_begin, add_end, add_begin_high, add_end_high, min_count, symbols_):
    #    return cls(bool(add_begin), bool(add_end), bool(add_begin_high), bool(add_end_high),
    #                int(min_count), StringList.from_string(symbols_))

    def __init__(self, add_begin: bool, add_end: bool,
                add_begin_high: bool, add_end_high: bool,
                min_count=1, symbols: Optional[list[str]] = None):
        super().__init__(add_begin, add_end, min_count, symbols)
        self.add_begin_high = add_begin_high
        self.add_end_high = add_end_high

    def vectorize_sequence(self, sequence: list[list[str]]) -> list[list[int]]:
        ans = [self.vectorize_element(element) for element in sequence]
        if self.add_begin_high:
            begin = list[int]()
            ans = [begin] + ans
        if self.add_end_high:
            end = list[int]()
            ans = ans + [end]
        return ans
