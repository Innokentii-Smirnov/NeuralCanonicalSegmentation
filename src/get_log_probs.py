from argparse import ArgumentParser
import os
from os import path
from typing import Callable, Sequence
import numpy as np
from torch import Tensor
from udapi.block.read.conllu import Conllu
from models.taggers.mc import SequenceClassifier, Sentence, InputWord
from library.dm import DM

def save_log_probs(log_probs: Sequence[Tensor], filename: str) -> None:
    array = np.array([tensor.cpu().numpy() for tensor in log_probs], dtype=object)
    with open(filename, 'wb') as fout:
        np.save(fout, array)

parser = ArgumentParser(prog='get_log_probs.py',
                        description='Get logarithms of label probabilities for each word in a CONLL-U corpus with several classifiers')
parser.add_argument('infile', help='A conllu file to analyze')
parser.add_argument('outdir', help='An name for the output directory')
parser.add_argument('model_directory', help='A directory containing the model vocabularies, a checkpoint and a list of ordered label fields')
parser.add_argument('attributes', help='The attributes to predict probability distributions for', nargs='*')
args = parser.parse_args()

for attribute in args.attributes:
  if not path.exists(path.join(args.model_directory, attribute)):
    print('No classifier was found for the attribute "{0}". Exiting.')
    exit()

reader = Conllu(files=[args.infile])
document = reader.read_documents()[0]

trees = [root for root in document.trees if len(root.descendants) > 0]

data = list[Sentence]()
for root in trees:
  sentence = Sentence()
  for node in root.descendants:
    form = node.form
    input_word: InputWord = {
      'exponent': form,
      'characters': list(form),
      'predet': node.misc['Det'],
      'postdet': node.misc['Postdet']
    }
    sentence.append(input_word)
  data.append(sentence)

os.makedirs(args.outdir, exist_ok=True)

with DM(args.outdir):
  for attribute in args.attributes:
    model = SequenceClassifier.load(path.join(args.model_directory, attribute), attribute)
    log_probs = model.get_log_probs_for(data)
    save_log_probs(log_probs, '{0}.npy'.format(attribute))
