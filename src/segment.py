from sys import argv
from os import path, listdir
from os.path import splitext
from utils.vocabulary import Vocabulary
from models.morphon import make_model
from utils.dataloader import DEVICE
import torch
if len(argv) != 5:
    print('The parameters are language, model type, model subtype, and the word')
    exit()
LANG, MODEL_TYPE, MODEL_SUBTYPE, word = argv[1:]
model_dir = path.join('models', LANG, MODEL_TYPE, MODEL_SUBTYPE)
vocab_dir = path.join(model_dir, 'Vocabularies')
vocabs = {splitext(filename)[0]: Vocabulary(True, True).load(path.join(vocab_dir, filename))
          for filename in listdir(vocab_dir)}
for key, vocab in vocabs.items():
    print(key, len(vocab))
model = make_model(MODEL_TYPE, MODEL_SUBTYPE, vocabs, DEVICE)
checkpoint_dir = path.join(model_dir, 'Checkpoints', '0')
checkpoint_file = path.join(checkpoint_dir, f'checkpoint_best_{MODEL_SUBTYPE}.pt')
model.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE))
result = model.apply([word])
for segmentation in result:
    print(segmentation)
