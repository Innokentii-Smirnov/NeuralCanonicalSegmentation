import argparse
from os import path, listdir
from os.path import splitext
from utils.vocabulary import Vocabulary
from models.morphon import make_model
from utils.dataloader import DEVICE
import torch
parser = argparse.ArgumentParser(
  prog='segment.py',
  description='Segment a word in the specified language with the specified model'
)
parser.add_argument('language', choices=listdir('models'),
                    help='the three-letter code of the language')
parser.add_argument('model_type', choices=['tagger', 'transducer'],
                    help='the general type of the model to use')
parser.add_argument('model_subtype', choices=['CNN', 'LSTM', 'RCNN', 'RCNN-skip-conn'],
                    help='the subtype of the model to use')
parser.add_argument('word',
                    help='the word to segment')
args = parser.parse_args()
model_dir = path.join('models', args.language, args.model_type, args.model_subtype)
vocab_dir = path.join(model_dir, 'Vocabularies')
vocabs = {splitext(filename)[0]: Vocabulary(True, True).load(path.join(vocab_dir, filename))
          for filename in listdir(vocab_dir)}
for key, vocab in vocabs.items():
    print(key, len(vocab))
model = make_model(args.model_type, args.model_subtype, vocabs, DEVICE)
checkpoint_dir = path.join(model_dir, 'Checkpoints', '0')
checkpoint_file = path.join(checkpoint_dir, f'checkpoint_best_{args.model_subtype}.pt')
model.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE))
result = model.apply([args.word])
for segmentation in result:
    print(segmentation)
