import os
from os import path
from typing import TypedDict
from collections import OrderedDict
from logging import getLogger
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm
from library.dm import DM
from arguments import EncoderArguments, CNNArguments
from encoders.cnn.two_level_sequential import TwoLevelSequentialEncoder
from decoders.mc import Mc
from utils.vocabulary import Vocabulary, SequenceVocabulary
from utils.dataset import SequenceDataset
from utils.dataloader import FieldBatchDataloader, DEVICE

class InputWord(TypedDict):
  exponent: str
  characters: list[str]
  predet: str
  postdet: str

Sentence = list[InputWord]

logger = getLogger(__name__)

class SequenceClassifier(nn.Module):

    def __init__(self, lower_encoder_arguments: CNNArguments,
                 lower_encoding_dropout: float,
                 higher_encoder_arguments: EncoderArguments,
                 features: OrderedDict[str, tuple[int, int, float]],
                 higher_encoding_dropout: float,
                 labels_number: int,
                 vocabularies: dict[str, Vocabulary],
                 sequence_vocabularies: dict[str, SequenceVocabulary],
                 target_attr: str):

        super(SequenceClassifier, self).__init__()
        self.encoder = TwoLevelSequentialEncoder(len(sequence_vocabularies['characters']),
                                                 lower_encoder_arguments,
                                                 lower_encoding_dropout,
                                                 higher_encoder_arguments,
                                                 features)
        self.higher_encoding_dropout = nn.Dropout(higher_encoding_dropout)
        self.decoder = Mc(self.encoder.output_dim, labels_number)
        self.vocabularies = vocabularies
        self.sequence_vocabularies = sequence_vocabularies
        self.target_attr = target_attr

    def forward(self, characters, **kwargs):
        embeddings = F.one_hot(characters, self.encoder.lower_encoder.input_dim).float()
        encoding = self.encoder(embeddings, {
            feature: kwargs[feature] for feature in self.encoder.higher_encoder.features
        })
        encoding = self.higher_encoding_dropout(encoding)
        return self.decoder(encoding)

    def predict(self, X: SequenceDataset):
        self.eval()
        dataloader = FieldBatchDataloader(X, batch_size=32)
        answer = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            with torch.no_grad():
                batch_answer = self(**batch)
            labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            #log_probs = batch_answer["log_probs"]
            #_, labels = torch.topk(log_probs, 10)
            for index, curr_labels, curr_mask in zip(indexes, labels, batch['mask'].bool().cpu().numpy(), strict=True):
                result = np.take(X.vocabs[self.target_attr].symbols_, curr_labels[curr_mask])
                answer[index] = result
        return answer

    def apply_to(self, data: list[Sentence]) -> list[list[str]]:
        dataset = SequenceDataset(data,
                          ['exponent', 'predet', 'postdet'],
                          ['characters'],
                          True, True, True, True,
                          self.vocabularies,
                          self.sequence_vocabularies)
        predictions = self.predict(dataset)
        return predictions

    @classmethod
    def initialize(cls, vocabularies: dict[str, Vocabulary],
                   sequence_vocabularies: dict[str, SequenceVocabulary],
                   target_attr: str) -> SequenceClassifier:
        model = cls(
          CNNArguments(n_layers=3, window=5, n_hidden=192, dropout=0.2, use_batch_norm=True),
          0.1,
          EncoderArguments(embedding_dim=0,
                          embedding_dropout=0,
                          hidden_size=256,
                          num_layers=1,
                          lstm_dropout=0.1,
                          bidirectional=True),
          OrderedDict({
              'exponent': (len(vocabularies["exponent"]), 128, 0.3),
              'predet': (len(vocabularies["predet"]), 64, 0.1),
              'postdet': (len(vocabularies["postdet"]), 64, 0.1)
          }),
          0.1,
          len(vocabularies[target_attr]),
          vocabularies,
          sequence_vocabularies,
          target_attr
        )
        return model

    @classmethod
    def load(cls, model_directory: str, target_attr: str) -> SequenceClassifier:
        with DM(model_directory):
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
        model = cls.initialize(vocabularies, sequence_vocabularies, target_attr)
        with DM(model_directory):
          checkpoint_path = path.join('Checkpoints', '0', 'checkpoint_best_Classifier.pt')
          model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        return model
