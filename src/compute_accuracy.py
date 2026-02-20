import os
from os import path
from library.read import read_list
import pandas as pd
from sklearn.metrics import accuracy_score
import json

LANGUAGE_CODES = ['deu', 'eng', 'ind']

CODE_TO_LANGUAGE = {
  'deu': 'german',
  'eng': 'english',
  'ind': 'indonesian'
}

MODEL_TYPE_TO_BOUNDARY = {
  'tagger': '@',
  'transducer': '-'
}

DATASET_PATH = 'canonical-segmentation'
CORR_FILE = 'test0'
PRED_DIR = 'predictions'
PRED_FILE = 'predictions.txt'
OUTDIR = 'metrics'
OUTFILE = 'Accuracy.json'

accuracies = dict[str, dict[str, float]]()

for code in LANGUAGE_CODES:
  accuracies[code] = dict[str, float]()
  language_dir = path.join(PRED_DIR, code)
  language = CODE_TO_LANGUAGE[code]
  corr_segm_dir = path.join(DATASET_PATH, language)
  corr_file = path.join(corr_segm_dir, CORR_FILE)
  daraFrame = pd.read_csv(corr_file, sep='\t', names=['orth', 'morphon', 'segm'])
  y_true = daraFrame['segm'].to_list()
  for model_type in os.listdir(language_dir):
    model_type_dir = path.join(language_dir, model_type)
    for model_subtype in os.listdir(model_type_dir):
      model_identifier = '{0}_{1}'.format(model_type, model_subtype)
      model_dir = path.join(model_type_dir, model_subtype)
      pred_file = path.join(model_dir, PRED_FILE)
      predictions = read_list(pred_file)
      boundary = MODEL_TYPE_TO_BOUNDARY[model_type]
      postprocess = lambda segm: segm.replace(boundary, ' ')
      y_pred = list(map(postprocess, predictions))
      accuracy: float = accuracy_score(y_true, y_pred)
      accuracies[code][model_identifier] = round(100 * accuracy, 2)

os.makedirs(OUTDIR, exist_ok=True)
outfile = path.join(OUTDIR, OUTFILE)
with open(outfile, 'w', encoding='utf-8') as fout:
  json.dump(accuracies, fout, ensure_ascii=False, indent='\t')
