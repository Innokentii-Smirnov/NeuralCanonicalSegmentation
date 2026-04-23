import argparse
import os
from os import path
import sys
from collections import namedtuple
import torch
from sklearn.metrics import accuracy_score
from reproducibility import set_seeds
from segm import load_data, prepare_checkpoints_dir, prepare_test, evaluate
from utils.dataloader import FieldBatchDataloader, DEVICE
from models.morphon import make_model
from trainers import make_trainer
from library.dm import DM
from training import do_epoch
from appliers import make_applier
from random import choices
from library.iterable import find
from library.align import align_and_stringify
from segm.input_word import parse_input_word
sys.path.insert(1, path.abspath('2022SegmentationST/evaluation'))
from evaluate import main as compute_sigmorphon_metrics

parser = argparse.ArgumentParser(
  prog='train.py',
  description='Train a neural model for canonical morpheme segmentation'
)
parser.add_argument('dataset',
                    help='the dataset on which to train the model')
parser.add_argument('language',
                    help='the three-letter code of the language')
parser.add_argument('--sep', default='~',
                    help='the separator for segmentation fragments used in the aligned data')
parser.add_argument('model_type', choices=['tagger', 'transducer', 'transformer'],
                    help='the general type of the model to use')
parser.add_argument('model_subtype', choices=['CNN', 'LSTM', 'RCNN', 'RCNN-skip-conn', 'char'],
                    help='the subtype of the model to use')
parser.add_argument('--epochs', type=int, default=50,
                    help='for how many epochs to train the model')
parser.add_argument('model_directory',
                    help='a directory to save the model and its vocabularies')
parser.add_argument('--load', action='store_true',
                    help='whether to load a model from an existing checkpoint')
parser.add_argument('--no-train', action='store_true',
                    help='only evaluate the model')
parser.add_argument('--use-features', action='store_true',
                    help='include morphological features as an additional input to the model')
parser.add_argument('words', nargs='*',
                    help='the words to segment (multiple values allowed)')
parser.add_argument('--num-encoder-layers', type=int,
                    help='number of layers in the encoder')
parser.add_argument('--num-decoder-layers', type=int,
                    help='number of layers in the decoder')
parser.add_argument('--emb-size', type=int,
                    help='embedding size')
parser.add_argument('--nhead', type=int,
                    help='number of heads for the tranformer')
parser.add_argument('--dim-feedforward', type=int,
                    help='feedforward layer dimension for the tranformer')
parser.add_argument('--dropout', type=float,
                    help='dropout value')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size')
parser.add_argument('--test_file', type=str,
                    help='a file containing the test data')
parser.add_argument('--pred_file', type=str,
                    help='a file to store the model\'s predictions')
parser.add_argument('--use-entmax', action='store_true',
                    help='use the entmax-1,5 activation and loss')
parser.add_argument('--use-general-attention', action='store_true',
                    help='use general attention instead of concat attention for LSTM transducer')
args = parser.parse_args()

test_file = args.test_file if args.test_file is not None \
  else path.join(args.dataset, f'{args.language}.word.test.gold.tsv')
pred_file = args.pred_file if args.pred_file is not None \
  else path.join(args.model_directory, f'{args.language}.word.test.predictions')

set_seeds()

aligned_data_required = (args.model_type == 'tagger')
X_train, X_dev = load_data(args.model_directory, args.dataset, args.language, args.sep,
                           aligned_data_required)

train_dataloader = FieldBatchDataloader(X_train)

checkpoint, checkpoints_dir, to_load, load_checkpoints_dir = prepare_checkpoints_dir(args.model_directory, args.model_subtype)

model = make_model(args.model_type, args.model_subtype, X_train.vocabs, DEVICE, args.use_features, args.model_directory, vars(args))
trainer = make_trainer(args.model_type, model, DEVICE)

if args.load and load_checkpoints_dir is not None:
  with DM(load_checkpoints_dir):
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

print(model)

vocab = X_train.vocabs["morphon"]
dev_dataloader = FieldBatchDataloader(X_dev, batch_size=args.batch_size, device=DEVICE)

if not args.no_train:
  best_val_acc = 0.0
  best_epoch = -1

  train_dataloader = FieldBatchDataloader(X_train, batch_size=args.batch_size, device=DEVICE)

  curr = to_load + 1
  curr_checkpoints_dir = path.join(checkpoints_dir, str(curr))
  os.makedirs(curr_checkpoints_dir, exist_ok=True)

  best_epoch = -1

  for epoch in range(args.epochs):
    do_epoch(trainer, train_dataloader, vocab, mode="train", epoch=epoch+1)
    epoch_metrics = do_epoch(trainer, dev_dataloader, vocab, mode="validate", epoch=epoch+1)
    if epoch_metrics.accuracy >= best_val_acc:
      best_val_acc = epoch_metrics.accuracy
      best_epoch = epoch
      with DM(curr_checkpoints_dir):
        torch.save(model.state_dict(), checkpoint)

  with DM(curr_checkpoints_dir):
    model.load_state_dict(torch.load(checkpoint, DEVICE))

  print(best_epoch)
  print(round(100 * best_val_acc, 2))

do_epoch(trainer, dev_dataloader, vocab, mode="validate", epoch="evaluate")

test_data, test_words, words_for_test, gold_segmentations = prepare_test(args.dataset, args.language)
applier = make_applier(args.model_type, model, X_train.vocabs, DEVICE)
segmentations = applier.apply_to(words_for_test, batch_size=args.batch_size)

if args.dataset == '2022SegmentationST/data':
  segmentations = [segmentation.replace('@', ' @@') for segmentation in segmentations]

with open(pred_file, 'w', encoding='utf-8') as fout:
  for (word, _), segmentation in zip(words_for_test, segmentations, strict=True):
    print(word, segmentation, sep='\t', file=fout)

for i in choices(list(range(len(words_for_test))), k=30):
    word, features = words_for_test[i]
    segmentation = segmentations[i]
    correct = gold_segmentations[i]
    correction = correct if segmentation != correct else ''
    print('{0:20} {1:20} {2}'.format(word, segmentation, correction))

evaluate(segmentations, gold_segmentations)

errors = [(word[0], word[1], segmentation) for word, segmentation
          in zip(test_data, segmentations, strict=True)
          if word[1] != segmentation]

print('Accuracy: {0} %, error rate: {1} / {2}.'.format(
  round(100 * ((len(test_words) - len(errors)) / len(test_words)), 2),
  len(errors), len(test_words)
))

errors.sort(key=lambda x: len(x[1]))

k = find(lambda i: len(errors[i][1]) >= 10, range(len(errors)))

print('Errors:')
if k is not None:
  print(align_and_stringify(errors[:k]))
  if k < len(errors) - 2:
    print(align_and_stringify(errors[k:-2]))
    print(align_and_stringify(errors[-2:]))
  else:
    print(align_and_stringify(errors[k:]))
else:
  print(align_and_stringify(errors))

for segmentation in applier.apply_to(list(map(parse_input_word, args.words))):
    print(segmentation)
print()

y_true = [segmentation for word, segmentation, features in test_data]
y_pred = segmentations
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: {:.2f}'.format(100 * accuracy))

Args = namedtuple('Args', ['gold', 'guess', 'category'])
compute_sigmorphon_metrics(Args(gold=test_file, guess=pred_file, category=False))
