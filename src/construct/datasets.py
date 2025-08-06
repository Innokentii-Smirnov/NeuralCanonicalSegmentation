import os
from os import path
from typing import Any
from library.dm import DM
from utils.vocabulary import Vocabulary, SequenceVocabulary
from utils.dataset import SequenceDataset


def make_datasets(X_train: SequenceDataset,
                  sentence_lists: list[list[dict[str, Any]]],
                  project_directory: str,
                  load_vocabularies: bool = False
                  ) -> list[SequenceDataset]:

    vdir = path.join(project_directory, 'Vocabularies',)
    lvdir = path.join(project_directory, 'Sequence vocabularies')

    if load_vocabularies:
        with DM(vdir):
            for filename in os.listdir(vdir):
                field = path.splitext(filename)[0]
                X_train.vocabs[field].load(field)
        with DM(lvdir):
            for filename in os.listdir(lvdir):
                field = path.splitext(filename)[0]
                X_train.seq_vocabs[field].load(field)

    else:
        X_train.fit_vocabs()
        os.makedirs(vdir, exist_ok=True)
        with DM(vdir):
            for field, vocab in X_train.vocabs.items():
                vocab.save(field)

        os.makedirs(lvdir, exist_ok=True)
        with DM(lvdir):
            for field, vocab in X_train.seq_vocabs.items():
                vocab.save(field)
    
    datasets = list[SequenceDataset]()
    
    for sentence_list in sentence_lists:
        dataset = X_train.alike(X_train, sentence_list)
        datasets.append(dataset)

    all_vocabs = X_train.vocabs | X_train.seq_vocabs
    for field, vocab in all_vocabs.items():
        print(field, len(vocab.symbols_))
    print()

    for field, vocab in all_vocabs.items():
        print(field, list(all_vocabs[field].symbols_[:10]))
    print()

    for field, elem in datasets[-1][4].items():
        print(field, elem)

    return datasets

    