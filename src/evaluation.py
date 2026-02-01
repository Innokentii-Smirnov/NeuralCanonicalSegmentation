from library.read import read_list
from typing import TypedDict
from models.morphon.transducer.basic import BasicMorphonologicalTransducer

class Word(TypedDict):
  phon: str
  morphon: list[str]

def read_data_all_corr(file: str) -> list[Word]:
    return [{'phon': token,
             'morphon': segmentations.split(', ')}
            for token, segmentations
             in map(lambda line: line.split('\t'), read_list(f'{LANG}.word.{file}.tsv'))]

def evaluate(segmentations: list[str], words: list[Word]):
  correct = 0
  for (segmentation, word) in zip(segmentations, words, strict=True):
    if segmentation in word['morphon']:
      correct += 1
  accuracy = 100 * correct / len(words)
  print('Accuracy: {0:.2f}'.format(accuracy))

def test(model: BasicMorphonologicalTransducer, words: list[Word]):
  input = [word['phon'] for word in words]
  segmentations = model.apply(input)
  evaluate(segmentations, words)
