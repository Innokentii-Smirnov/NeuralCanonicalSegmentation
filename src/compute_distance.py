from itertools import starmap
from statistics import fmean
from metric_computation import compute_metrics, MetricFunction
import sys
sys.path.insert(1, '2022SegmentationST/evaluation')
from evaluate import distance

LANGUAGE_CODES = ['xhu', 'ces']

def preprocess(segmentation: str) -> str:
  return segmentation.replace(' @@', '|')

def mean_levenshtein_distance(y_true: list[str], y_pred: list[str]) -> float:
  return fmean(starmap(distance, zip(map(preprocess, y_true),
                                     map(preprocess, y_pred), strict=True)))

METRIC_FUNCTION: MetricFunction = lambda y_true, y_pred: \
  {'distance': mean_levenshtein_distance(y_true, y_pred)}

compute_metrics(METRIC_FUNCTION, LANGUAGE_CODES)
