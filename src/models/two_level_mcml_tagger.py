import os
from os import path
from typing import TypedDict
from collections import OrderedDict
from logging import getLogger
from tqdm.auto import tqdm
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, Dropout
import torch.nn.functional as F
from arguments import EncoderArguments, CNNArguments
from encoders.cnn.two_level_sequential import TwoLevelSequentialEncoder
from decoders.mcml import McMl
from library.dm import DM
from library.read import read_list
from utils.vocabulary import Vocabulary, SequenceVocabulary
from utils.dataloader import FieldBatchDataloader, DEVICE
from utils.dataset import SequenceDataset

class InputWord(TypedDict):
  word: str
  letters: list[str]
  det: str
  postdet: str

Sentence = list[InputWord]

LabelDict = dict[str, str]

logger = getLogger(__name__)

MISSING = '<NO>'

class TwoLevelMcMlTagger(Module):

    def __init__(self,
                 letter_vocab_size: int,
                 lower_encoder_arguments: CNNArguments,
                 lower_encoding_dropout: float,
                 higher_encoder_arguments: EncoderArguments,
                 features: OrderedDict[str, tuple[int, int, float]],
                 higher_encoding_dropout: float,
                 labels_number: OrderedDict[str, int],
                 vocabularies: dict[str, Vocabulary],
                 sequence_vocabularies: dict[str, SequenceVocabulary]):
        super(TwoLevelMcMlTagger, self).__init__()
        self.encoder = TwoLevelSequentialEncoder(letter_vocab_size,
                                                 lower_encoder_arguments,
                                                 lower_encoding_dropout,
                                                 higher_encoder_arguments,
                                                 features)
        self.higher_encoding_dropout = Dropout(higher_encoding_dropout)
        self.decoder = McMl(labels_number, self.encoder.output_dim)
        self.label_fields: list[str] = list(labels_number)
        self.hit_encls_fields = self.label_fields[4:]
        self.vocabularies = vocabularies
        self.sequence_vocabularies = sequence_vocabularies

    def forward(self, letters, **kwargs):
        encoding = self.encoder(
            F.one_hot(
                letters,
                self.encoder.lower_encoder.input_dim
            ).float(), {
            feature: kwargs[feature] for feature in self.encoder.higher_encoder.features
        })
        encoding = self.higher_encoding_dropout(encoding)
        return self.decoder(encoding)

    def get_label(self, d: dict[str, str]) -> str:
        label = d['word-label']
        for field in ['relator-1', 'relator-2']:
            if d[field] != MISSING:
                label += '.' + d[field]
        if d['akk-poss'] != MISSING:
            label += '_' + d['akk-poss']
        hit_encls = [d[field] for field in self.hit_encls_fields if d[field] != MISSING]
        if len(hit_encls) > 0:
            label += '=' + '='.join(hit_encls)
        return label

    def predict(self, dataloader: FieldBatchDataloader):
        X = dataloader.X
        self.eval()
        answer = list[list[dict[str, str]]]()
        for i in range(len(X)):
            d = list[dict[str, str]]()
            answer.append(d)
        for batch in tqdm(dataloader):
            with torch.no_grad():
                batch_answer = self(**batch)
            labels_dict: dict[str, Tensor] = batch_answer["labels"]
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for field in self.label_fields:
                labels = labels_dict[field].cpu().numpy()
                for index, curr_labels, curr_mask in zip(batch["indexes"], labels, batch["mask"].bool().cpu().numpy()):
                    result = np.take(X.vocabs[field].symbols_, curr_labels[curr_mask])
                    if field == 'word-label':
                        for i in range(len(result)):
                            answer[index].append(dict[str, str]())
                    for i in range(len(result)):
                        answer[index][i][field] = result[i]
        for index in range(len(answer)):
            for i in range(len(answer[index])):
                label = self.get_label(answer[index][i])
                answer[index][i]['label'] = label
        return answer

    def apply_to(self, data: list[Sentence]) -> list[list[LabelDict]]:
        dataset = SequenceDataset(data,
                          ['word', 'det', 'postdet'],
                          ['letters'],
                          True, True, True, True,
                          self.vocabularies,
                          self.sequence_vocabularies)
        dataloader = FieldBatchDataloader(dataset)

        predictions = self.predict(dataloader)
        return predictions

    @classmethod
    def initialize(cls, vocabularies: dict[str, Vocabulary],
                   sequence_vocabularies: dict[str, SequenceVocabulary],
                   label_fields: list[str]) -> TwoLevelMcMlTagger:
        model = cls(
          len(sequence_vocabularies['letters']),
          CNNArguments(n_layers=3, window=5, n_hidden=192, dropout=0.2, use_batch_norm=True),
          0.1,
          EncoderArguments(embedding_dim=0,
                          embedding_dropout=0,
                          hidden_size=256,
                          num_layers=1,
                          lstm_dropout=0.1,
                          bidirectional=True),
          OrderedDict({
              #'word': (len(vocabs["word"]), 128, 0.3),
              'det': (len(vocabularies["det"]), 64, 0.1),
              'postdet': (len(vocabularies["postdet"]), 64, 0.1)
          }),
          0.1,
          OrderedDict({field: len(vocabularies[field].symbols_) for field in label_fields}),
          vocabularies,
          sequence_vocabularies
        )
        return model

    @classmethod
    def load(cls, model_directory: str) -> TwoLevelMcMlTagger:
        with DM(model_directory):
          label_fields = read_list('Label_fields.txt')
          with DM('Vocabularies'):
            vocabularies = {
              path.splitext(filename)[0]: Vocabulary(True, True).load(filename)
              for filename in os.listdir('.')
            }
          with DM('Sequence vocabularies'):
            sequence_vocabularies = {
              path.splitext(filename)[0]: SequenceVocabulary(True, True, True, True).load(filename)
              for filename in os.listdir('.')
            }
        all_vocabularies = vocabularies | sequence_vocabularies
        for key, vocabulary in all_vocabularies.items():
          logger.info('%s %i', key, len(vocabulary))

        model = cls.initialize(vocabularies, sequence_vocabularies, label_fields)
        with DM(model_directory):
          model.load_state_dict(torch.load('checkpoint.pt', map_location=DEVICE))
        return model
