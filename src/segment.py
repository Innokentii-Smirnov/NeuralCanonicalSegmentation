from sys import argv
from os import path, listdir
from os.path import splitext
from utils.vocabulary import Vocabulary
from models.morphon.tagger import make_model
from utils.dataloader import DEVICE
import torch
if len(argv) != 4:
    print('The parameters are language, model and the word')
    exit()
LANG, MODEL, word = argv[1:]
model_dir = path.join('models', LANG, MODEL)
vocab_dir = path.join(model_dir, 'Vocabularies')
vocabs = {splitext(filename)[0]: Vocabulary(True, True).load(path.join(vocab_dir, filename))
          for filename in listdir(vocab_dir)}
for key, vocab in vocabs.items():
    print(key, len(vocab))
model = make_model(MODEL, vocabs, DEVICE)
checkpoint_dir = path.join(model_dir, 'Checkpoints', '0')
checkpoint_file = path.join(checkpoint_dir, f'checkpoint_best_{MODEL}.pt')
model.load_state_dict(torch.load(checkpoint_file, map_location=DEVICE))
result = model.apply([word])
for segmentation in result:
    print(segmentation)
