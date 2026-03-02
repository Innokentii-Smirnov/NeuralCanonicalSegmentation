import os
from os import path
from library.read import read_list
import pandas as pd
from sklearn.metrics import accuracy_score
import json

CS = 'canonical-segmentation'
SM = '2022SegmentationST'
OTHER = 'data'

LANGUAGE_CODES = ['deu', 'eng', 'ind', 'ita', 'fra', 'lat', 'chu', 'vsn']

CODE_TO_LANGUAGE = {
  'deu': 'german',
  'eng': 'english',
  'ind': 'indonesian'
}

MODEL_TYPE_TO_BOUNDARY = {
  'tagger': '@',
  'transducer': '-'
}

LANGUAGE_TO_DATASET = {
  'deu': CS,
  'eng': CS,
  'ind': CS,
  'ita': SM,
  'fra': SM,
  'lat': SM,
  'chu': OTHER,
  'vsn': OTHER
}

DATASET_TO_BOUNDARY = {
  CS: ' ',
  SM: ' @@'
}

DATASET_TO_INFILE = {
  CS: lambda code: path.join(CS, CODE_TO_LANGUAGE[code], 'test0'),
  SM: lambda code: path.join(SM, 'data', '{0}.word.test.gold.tsv'.format(code)),
  OTHER: lambda code: path.join(OTHER, '{0}.word.test.gold.tsv'.format(code))
}

DATASET_TO_COLUMNS = {
  CS: ['orth', 'morphon', 'segm'],
  SM: ['orth', 'segm'],
  OTHER: ['orth', 'segm']
}

PRED_DIR = 'predictions'
PRED_FILE = 'predictions.txt'
OUTDIR = 'metrics'
OUTFILE = 'Accuracy.json'

accuracies = dict[str, dict[str, float]]()

for code in LANGUAGE_CODES:
  accuracies[code] = dict[str, float]()
  dataset = LANGUAGE_TO_DATASET[code]
  language_dir = path.join(PRED_DIR, code)
  corr_file = DATASET_TO_INFILE[dataset](code)
  columns = DATASET_TO_COLUMNS[dataset]
  dataFrame = pd.read_csv(corr_file, sep='\t', names=columns,
                          usecols=list(range(len(columns))))
  y_true = dataFrame['segm'].to_list()
  for model_type in os.listdir(language_dir):
    model_type_dir = path.join(language_dir, model_type)
    for model_subtype in os.listdir(model_type_dir):
      model_identifier = '{0}_{1}'.format(model_type, model_subtype)
      model_dir = path.join(model_type_dir, model_subtype)
      pred_file = path.join(model_dir, PRED_FILE)
      predictions = read_list(pred_file)
      if dataset in DATASET_TO_BOUNDARY:
        pred_boundary = MODEL_TYPE_TO_BOUNDARY[model_type]
        dataset_boundary = DATASET_TO_BOUNDARY[dataset]
        postprocess = lambda segm: segm.replace(pred_boundary, dataset_boundary)
        y_pred = list(map(postprocess, predictions))
      else:
        y_pred = predictions
      accuracy: float = accuracy_score(y_true, y_pred)
      accuracies[code][model_identifier] = round(100 * accuracy, 2)

os.makedirs(OUTDIR, exist_ok=True)
outfile = path.join(OUTDIR, OUTFILE)
with open(outfile, 'w', encoding='utf-8') as fout:
  json.dump(accuracies, fout, ensure_ascii=False, indent='\t')
