from argparse import ArgumentParser
from os import path
from typing import Callable
from udapi.block.read.conllu import Conllu
from udapi.core.node import Node
from models.taggers.mc import SequenceClassifier, Sentence, InputWord

parser = ArgumentParser(prog='tag_and_lemmatize.py',
                        description='Morphologically tag and lemmatize a corpus in a conllu file')
parser.add_argument('infile', help='A conllu file to tag')
parser.add_argument('outfile', help='An name for the resulting conllu')
parser.add_argument('model_directory', help='A directory containing the model vocabularies, a checkpoint and a list of ordered label fields')
args = parser.parse_args()

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

def set_lemma(node: Node, lemma: str) -> None:
  node.lemma = lemma

def set_upos(node: Node, upos: str) -> None:
  node.upos = upos

def set_gloss(node: Node, gloss: str) -> None:
  node.gloss = gloss

def set_gramm_form(node: Node, gramm_form: str) -> None:
  node.xpos = gramm_form

def set_encl_chain(node: Node, encl_chain: str) -> None:
  if encl_chain != '<NO>':
    node.misc['encl_chain'] = encl_chain

fields: list[tuple[str, Callable[[Node, str], None]]] = [
  ('lemma', set_lemma),
  ('upos', set_upos),
  ('gloss', set_gloss),
  ('gramm_form', set_gramm_form),
  ('encl_chain', set_encl_chain)
]

for attribute, setter_function in fields:

  model = SequenceClassifier.load(path.join(args.model_directory, attribute), attribute)

  predictions = model.apply_to(data)

  for root, pred_values in zip(trees, predictions, strict=True):
    for node, pred_value in zip(root.descendants, pred_values, strict=True):
      setter_function(node, pred_value)

document.store_conllu(args.outfile)
