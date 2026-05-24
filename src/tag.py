from argparse import ArgumentParser
from udapi.block.read.conllu import Conllu
from models.two_level_mcml_tagger import TwoLevelMcMlTagger, Sentence, InputWord

parser = ArgumentParser(prog='tag.py',
                        description='Morphologically tag a corpus in a conllu file')
parser.add_argument('infile', help='A conllu file to tag')
parser.add_argument('outfile', help='An name for the resulting conllu')
parser.add_argument('model_directory', help='A directory containing the model vocabularies, a checkpoint and a list of ordered label fields')
args = parser.parse_args()

model = TwoLevelMcMlTagger.load(args.model_directory)

reader = Conllu(files=[args.infile])
document = reader.read_documents()[0]

trees = [root for root in document.trees if len(root.descendants) > 0]

data = list[Sentence]()
for root in trees:
  sentence = Sentence()
  for node in root.descendants:
    form = node.form
    input_word: InputWord = {
      'word': form,
      'letters': list(form),
      'det': node.misc['Det'],
      'postdet': node.misc['Postdet']
    }
    sentence.append(input_word)
  data.append(sentence)

predictions = model.apply_to(data)

for root, pred_label_dicts in zip(trees, predictions, strict=True):
  for node, pred_label_dict in zip(root.descendants, pred_label_dicts, strict=True):
    node.xpos = pred_label_dict['label']

document.store_conllu(args.outfile)
