import os
from os import path
from collections import defaultdict
from itertools import starmap
from typing import Callable
from statistics import fmean
from re import compile, sub
from library.read import read_list
import pandas as pd
import json

MetricFunction = Callable[[list[str], list[str]], dict[str, float]]

CS = 'canonical-segmentation'
SM = '2022SegmentationST'
OTHER = 'data'
HURRIAN = 'Hurrian'

CODE_TO_LANGUAGE = {
  'deu': 'german',
  'eng': 'english',
  'ind': 'indonesian'
}

MODEL_TYPE_TO_BOUNDARY = {
  'tagger': compile('@'),
  'transducer': compile('[-@]'),
  'transformer': compile('-')
}

LANGUAGE_TO_DATASET = {
  'deu': CS,
  'eng': CS,
  'ind': CS,
  'ita': SM,
  'fra': SM,
  'lat': SM,
  'chu': OTHER,
  'vsn': OTHER,
  'xhu': HURRIAN,
  'ces': SM
}

DATASET_TO_BOUNDARY = {
  CS: ' ',
  SM: ' @@'
}

DATASET_TO_INFILE = {
  CS: lambda code: path.join(CS, CODE_TO_LANGUAGE[code], 'test0'),
  SM: lambda code: path.join(SM, 'data', '{0}.word.test.gold.tsv'.format(code)),
  OTHER: lambda code: path.join(OTHER, '{0}.word.test.gold.tsv'.format(code)),
  HURRIAN: lambda code: path.join(HURRIAN, '{0}.word.test.gold.tsv'.format(code))
}

DATASET_TO_COLUMNS = {
  CS: ['orth', 'morphon', 'segm'],
  SM: ['orth', 'segm'],
  OTHER: ['orth', 'segm'],
  HURRIAN: ['orth', 'segm', 'pos'],
}

PRED_DIR = 'predictions'
PRED_FILE = 'predictions.txt'
OUTDIR = 'metrics'
OUTFILES = {
  'accuracy': 'Accuracy.json',
  'distance': 'Distance.json',
  'precision': 'Precision.json', 'recall': 'Recall.json', 'f_measure': 'F1.json'
}

def compute_metrics(metric_function: MetricFunction, language_codes: list[str],
                    round_ndigits: int = 2) -> None:
  metrics = defaultdict[str, defaultdict[str, dict[str, float]]](lambda: defaultdict(dict))

  for code in language_codes:
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
        if dataset in DATASET_TO_BOUNDARY and model_type in MODEL_TYPE_TO_BOUNDARY:
          pred_boundary = MODEL_TYPE_TO_BOUNDARY[model_type]
          dataset_boundary = DATASET_TO_BOUNDARY[dataset]
          postprocess = lambda segm: sub(pred_boundary, dataset_boundary, segm)
          y_pred = list(map(postprocess, predictions))
        else:
          y_pred = predictions
        computed = metric_function(y_true, y_pred)
        for metric, value in computed.items():
          metrics[metric][code][model_identifier] = round(value, round_ndigits)

  os.makedirs(OUTDIR, exist_ok=True)
  for metric, values in metrics.items():
    outfile = path.join(OUTDIR, OUTFILES[metric])
    with open(outfile, 'w', encoding='utf-8') as fout:
      json.dump(values, fout, ensure_ascii=False, indent='\t')
