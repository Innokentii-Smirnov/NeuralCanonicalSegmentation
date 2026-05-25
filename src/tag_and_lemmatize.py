from argparse import ArgumentParser
from os import path
from udapi.block.read.conllu import Conllu
from models.taggers.mc import SequenceClassifier, Sentence, InputWord

parser = ArgumentParser(prog='tag_and_lemmatize.py',
                        description='Morphologically tag and lemmatize a corpus in a conllu file')
parser.add_argument('infile', help='A conllu file to tag')
parser.add_argument('outfile', help='An name for the resulting conllu')
parser.add_argument('model_directory', help='A directory containing the model vocabularies, a checkpoint and a list of ordered label fields')
args = parser.parse_args()

lemmatizer = SequenceClassifier.load(path.join(args.model_directory, 'lemma'), 'lemma')

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

predictions = lemmatizer.apply_to(data)

for root, pred_lemmata in zip(trees, predictions, strict=True):
  for node, pred_lemma in zip(root.descendants, pred_lemmata, strict=True):
    node.lemma = pred_lemma

document.store_conllu(args.outfile)
