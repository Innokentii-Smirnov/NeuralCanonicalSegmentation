import argparse
from os import path, listdir
from os.path import splitext
from utils.vocabulary import Vocabulary
from models.morphon import make_model
from utils.dataloader import DEVICE
import torch
import logging
logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
  prog='segment.py',
  description='Segment words in the specified language with the specified model',
  epilog='If no input words are provided as arguments, words are read from the standard input until an empty line is encountered'
)
parser.add_argument('language', choices=listdir('models'),
                    help='the three-letter code of the language')
parser.add_argument('model_type', choices=['tagger', 'transducer'],
                    help='the general type of the model to use')
parser.add_argument('model_subtype', choices=['CNN', 'LSTM', 'RCNN', 'RCNN-skip-conn'],
                    help='the subtype of the model to use')
parser.add_argument('words', nargs='*',
                    help='the words to segment (multiple values allowed)')
args = parser.parse_args()
model_dir = path.join('models', args.language, args.model_type, args.model_subtype)
vocab_dir = path.join(model_dir, 'Vocabularies')
vocabs = {splitext(filename)[0]: Vocabulary(True, True).load(path.join(vocab_dir, filename))
          for filename in listdir(vocab_dir)}
for key, vocab in vocabs.items():
  logging.info('%s %i', key, len(vocab))
model = make_model(args.model_type, args.model_subtype, vocabs, DEVICE)
checkpoint_dir = path.join(model_dir, 'Checkpoints', '0')
checkpoint_file = path.join(checkpoint_dir, f'checkpoint_best_{args.model_subtype}.pt')
model.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE))
if len(args.words) > 0:
  words = args.words
else:
  words = list[str]()
  try:
    while (word := input()) != '':
      words.append(word)
  except EOFError:
    pass
result = model.apply(words)
for segmentation in result:
    print(segmentation)
