import os
from os import path
from random import choice, choices
from alignment import align
from stringutils import string_to_list
from utils.dataset import SimpleDataset
from construct.datasets import make_datasets
from library.constants import test_random_state, dev_random_state
from library.dm import DM
from sklearn.model_selection import train_test_split

FEATURE_SEP = ';'

def load_pairs(filename: str, sep: str, aligned: bool) -> list[tuple[list[str], list[str], list[str] | None]]:
    data = list[tuple[list[str], list[str], list[str] | None]]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            match line.split('\t'):
              case word, segmentation, joined_features:
                features: list[str] | None = joined_features.split(FEATURE_SEP)
              case word, segmentation:
                features = None
            elements = string_to_list(word, True)
            if aligned:
              labels = segmentation.split(sep)
              assert len(elements) == len(labels), (elements, labels)
            else:
              labels = string_to_list(segmentation, True)
            data.append((elements, labels, features))
    return data

def get_words(data):
    return [{'phon': phon, 'morphon': morphon, 'features': features} for phon, morphon, features in data]

def get_test_words(data):
    return [{'phon': phon, 'features': features} for phon, features in data]

def load_data(model_directory: str,
              dataset: str, lang: str, sep: str,
              aligned: bool,
              alignment_algorithm: str = 'Levenshtein'):

    os.makedirs(model_directory, exist_ok=True)

    if aligned:
      data_folder = path.join('aligned_data', alignment_algorithm, dataset)
    else:
      data_folder = dataset
    assert path.exists(data_folder), data_folder
    with DM(data_folder):
      train_data = load_pairs(f'{lang}.word.train.tsv', sep, aligned)
      dev_data = load_pairs(f'{lang}.word.dev.tsv', sep, aligned)

    for word, segmentation, _ in choices(train_data, k=20):
      if aligned:
        print(align((word, segmentation)))
      else:
        print(''.join(word), ''.join(segmentation))
        print()

    for part in train_data, dev_data:
        print(len(part))

    for x, y, f in choices(train_data, k=5):
        print('{0:32}{1:32}{2}'.format(''.join(x), ''.join(y), f))
    print()

    for x, y, f in choices(dev_data, k=5):
        print('{0:32}{1:32}{2}'.format(''.join(x), ''.join(y), f))
    print()

    train_words = get_words(train_data)
    dev_words = get_words(dev_data)

    for elem in train_words[:5]:
        print(elem)

    longest = max(map(lambda x: x['morphon'], train_words), key=len)
    print(''.join(longest))
    max_sequence_length = len(longest)
    print(max_sequence_length)

    mask_field = 'phon' if aligned else 'morphon'
    X_train = SimpleDataset(train_words, ['phon', 'morphon', 'features'], [], True, True, True, True, mask_field=mask_field)
    X_dev, = make_datasets(
        X_train,
        [dev_words],
        model_directory
    )

    difference = set(X_train.vocabs['morphon'].symbols_) - set(X_train.vocabs['phon'].symbols_)
    print(len(difference))
    for symbol in difference:
        print(symbol)

    for key, value in choice(train_words).items():
        print(key, ''.join(value) if isinstance(value, list) else value)
    return X_train, X_dev

def prepare_checkpoints_dir(model_directory: str, model_subtype: str):
    checkpoints_dir = path.join(model_directory, 'Checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint = "checkpoint_best_{0}.pt".format(model_subtype)

    files = os.listdir(checkpoints_dir)
    if len(files) > 0:
        to_load = max(map(int, files))
        load_checkpoints_dir = path.join(checkpoints_dir, str(to_load))
    else:
        to_load = -1
        load_checkpoints_dir = None
    return checkpoint, checkpoints_dir, to_load, load_checkpoints_dir

def load_test_words(filename: str) -> list[tuple[str, list[str] | None]]:
    data = list[tuple[str, list[str] | None]]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            match line.split('\t'):
              case word, joined_features:
                features: list[str] | None = joined_features.split(FEATURE_SEP)
              case word,:
                features = None
            data.append((word, features))
    return data

def load_test_data(filename: str) -> list[tuple[str, str, list[str] | None]]:
    data = list[tuple[str, str, list[str] | None]]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            match line.split('\t'):
              case word, segmentation, joined_features:
                features: list[str] | None = joined_features.split(FEATURE_SEP)
              case word, segmentation:
                features = None
            data.append((word, segmentation, features))
    return data

def prepare_test(dataset: str, lang: str):
    original_folder = dataset
    assert path.exists(original_folder), original_folder
    with DM(original_folder):
      words_for_test = load_test_words(f'{lang}.word.test.tsv')
      test_data = load_test_data(f'{lang}.word.test.gold.tsv')
    test_words = get_test_words(words_for_test)
    gold_segmentations = [segm for _, segm, _ in test_data]
    return test_data, test_words, words_for_test, gold_segmentations

def evaluate(predictions, gold_segmentations):
    correct = 0
    for prediction, gold_segmentation in zip(predictions, gold_segmentations, strict=True):
        if prediction == gold_segmentation:
            correct += 1
    accuracy = 100 * correct / len(gold_segmentations)
    print('Accuracy: {0} % ({1} / {2})'.format(round(accuracy, 2), correct, len(gold_segmentations)))
