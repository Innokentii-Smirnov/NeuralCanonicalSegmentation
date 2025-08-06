import os
from os import path
from random import choice, choices
from alignment import align
from stringutils import string_to_list
from utils.dataset import SimpleDataset
from construct.datasets import make_datasets
from library.constants import test_random_state, dev_random_state
from sklearn.model_selection import train_test_split

SEP = ':'
folder = '/content/drive/MyDrive/6-Segmentation'

def load_pairs(filename: str) -> list[tuple[list[str], list[str]]]:
    data = list[tuple[list[str], list[str]]]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            word, segmentation = line.split('\t')
            elements = string_to_list(word)
            labels = segmentation.split(SEP)
            assert len(elements) == len(labels), (elements, labels)
            data.append((elements, labels))
    return data

def get_words(data):
    return [{'phon': phon, 'morphon': morphon} for phon, morphon in data]

def get_test_words(data):
    return [{'phon': phon} for phon in data]

def load_data(directory: str, lang: str, model: str):
    global LANG, MODEL, NAME
    LANG = lang
    MODEL = model
    NAME = LANG + '/' + MODEL
    assert path.exists(folder)
    directory = path.join(folder, NAME)
    aligned_folder = f'{folder}/Data/{directory}/Aligned'
    assert path.exists(aligned_folder), aligned_folder
    os.chdir(aligned_folder)
    train_data = load_pairs(f'{LANG}.word.train.tsv')
    dev_data = load_pairs(f'{LANG}.word.dev.tsv')

    for elem in choices(train_data, k=20):
        print(align(elem))
        print()

    for part in train_data, dev_data:
        print(len(part))

    for x, y in choices(train_data, k=5):
        print('{0:32}{1}'.format(''.join(x), ''.join(y)))
    print()

    for x, y in choices(dev_data, k=5):
        print('{0:32}{1}'.format(''.join(x), ''.join(y)))
    print()

    train_words = get_words(train_data)
    dev_words = get_words(dev_data)

    for elem in train_words[:5]:
        print(elem)

    longest = max(map(lambda x: x['morphon'], train_words), key=len)
    print(''.join(longest))
    max_sequence_length = len(longest)
    print(max_sequence_length)

    X_train = SimpleDataset(train_words, ['phon', 'morphon'], [], True, True, True, True)
    X_dev, = make_datasets(
        X_train,
        [dev_words],
        directory
    )

    difference = set(X_train.vocabs['morphon'].symbols_) - set(X_train.vocabs['phon'].symbols_)
    print(len(difference))
    for symbol in difference:
        print(symbol)

    for key, value in choice(train_words).items():
        print(key, ''.join(value))
    return X_train, X_dev

def prepare_checkpoints_dir():
    corpus_directory = path.join(folder, NAME)
    checkpoints_dir = path.join(corpus_directory, 'Checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint = "checkpoint_best_{0}.pt".format(MODEL)

    files = os.listdir(checkpoints_dir)
    if len(files) > 0:
        to_load = max(map(int, files))
        load_checkpoints_dir = path.join(checkpoints_dir, str(to_load))
    else:
        to_load = -1
        load_checkpoints_dir = None
    return checkpoint, checkpoints_dir, to_load, load_checkpoints_dir

def load_test_words(filename: str) -> list[str]:
    data = list[str]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            data.append(line)
    return data

def load_test_data(filename: str) -> list[tuple[str, str]]:
    data = list[tuple[str, str]]()
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.rstrip()
            word, segmentation = line.split('\t')[:2]
            data.append((word, segmentation))
    return data

def prepare_test(LANG):
    original_folder = f'{folder}/Data/Sigmorphon 2022/Original'
    assert path.exists(original_folder), original_folder
    os.chdir(original_folder)
    words_for_test = load_test_words(f'{LANG}.word.test.tsv')
    test_words = get_test_words(words_for_test)
    test_data = load_test_data(f'{LANG}.word.test.gold.tsv')
    gold_segmentations = [segm for _, segm in test_data]
    return test_data, test_words, words_for_test, gold_segmentations

def evaluate(predictions, gold_segmentations):
    correct = 0
    for prediction, gold_segmentation in zip(predictions, gold_segmentations, strict=True):
        if prediction == gold_segmentation:
            correct += 1
    accuracy = 100 * correct / len(gold_segmentations)
    print('Accuracy: {0} % ({1} / {2})'.format(round(accuracy, 2), correct, len(gold_segmentations)))
