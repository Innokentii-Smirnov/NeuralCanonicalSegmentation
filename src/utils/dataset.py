from torch.utils.data import Dataset
from library.iterable import chain_seq
from .vocabulary import Vocabulary, SequenceVocabulary
from itertools import zip_longest, chain
from typing import Optional

class SequenceDataset(Dataset):

    def __init__(self, data, fields: list[str], list_fields: list[str], add_begin: bool, add_end: bool,
                 add_begin_low: bool, add_end_low: bool,
                 vocabs: Optional[dict[str, Vocabulary]] = None,
                 seq_vocabs: Optional[dict[str, Vocabulary]] = None):
        """
            data: List[List[dict]],
                список предложений,
                каждое предложение -- список слов,
                каждое слово -- словарь вида {"word": word, "tag": tag, "label": label, ...}
        """
        self.data = data
        self.fields = fields
        self.list_fields = list_fields
        self.add_begin = add_begin
        self.add_end = add_end
        self.add_begin_low = add_begin_low
        self.add_end_low = add_end_low
        if vocabs is None:
            assert seq_vocabs is None
            self.create_vocabs()
        else:
            self.save_vocabs(vocabs, seq_vocabs)

    def _make_mask(self, item):
        return self.vocabs[self.fields[0]].make_mask(item)
    
    def get_field(self, index: int, field: str) -> list[str]:
        return [elem[field] for elem in self.data[index]]

    def __getitem__(self, index):
        answer = dict()
        for field in self.fields:
            vocab = self.vocabs[field]
            # elem = {"word": ..., "tag": ..., "label": ...}
            answer[field] = vocab.vectorize_element(self.get_field(index, field))
        for field in self.list_fields:
            seq_vocab = self.seq_vocabs[field]
            answer[field] = seq_vocab.vectorize_sequence(self.get_field(index, field))
        # маска нужна, чтобы потом отличать позиции реальных ответов от позиций паддинга и вспомогательных символов
        answer["mask"] = self._make_mask(self.data[index])
        return answer

    def __len__(self):
        return len(self.data)

    def create_vocabs(self):
        self.vocabs = dict[str, Vocabulary]()
        for field in self.fields:
            vocab = Vocabulary(self.add_begin, self.add_end)
            self.vocabs[field] = vocab
        self.seq_vocabs = dict[str, SequenceVocabulary]()
        for field in self.list_fields:
            vocab = SequenceVocabulary(self.add_begin_low, self.add_end_low, self.add_begin, self.add_end)
            self.seq_vocabs[field] = vocab
    
    def fit_vocabs(self):
        for field in self.fields:
            data_for_vocab = (self.get_field(index, field) for index in range(len(self.data)))
            self.vocabs[field].fit(data_for_vocab)
        for field in self.list_fields:
            data_for_vocab = chain_seq(self.get_field(index, field) for index in range(len(self.data)))
            self.seq_vocabs[field].fit(data_for_vocab)

    def save_vocabs(self, vocabs: dict[str, Vocabulary], seq_vocabs: dict[str, Vocabulary]):
        self.vocabs = vocabs
        self.seq_vocabs = seq_vocabs

    @classmethod
    def alike(cls, other, data, fields: Optional[list[str]] = None, list_fields: Optional[list[str]] = None):
        if fields is None:
            fields = other.fields
        if list_fields is None:
            list_fields = other.list_fields
        dataset = cls(
            data, fields, list_fields,
            other.add_begin, other.add_end,
            other.add_begin_low, other.add_end_low,
            other.vocabs, other.seq_vocabs
        )
        return dataset
    
class SimpleDataset(SequenceDataset):

    def _make_mask(self, item: dict[str, list[str]]) -> list[bool]:
        return super()._make_mask(item[self.fields[0]])
    
    def get_field(self, index: int, field: str) -> list[str]:
        return self.data[index][field]
