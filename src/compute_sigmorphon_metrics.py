import re
from metric_computation import compute_metrics, MetricFunction
import sys
sys.path.insert(1, '2022SegmentationST/evaluation')
from evaluate import distance, n_correct, compute_stats

LANGUAGE_CODES = ['xhu', 'ces']

MORPHEME_BOUNDARY = re.compile(r'(?<=[.+-=])|\|')

def preprocess(segmentation: str) -> str:
  return segmentation.replace(' @@', '|')

def preprocess_for_n_correct(segmentation: str) -> str:
  return '|'.join(MORPHEME_BOUNDARY.split(segmentation))

def metric_function(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
  gold_data = list(map(preprocess, y_true))
  guess_data = list(map(preprocess, y_pred))

  # levenshtein distance can be computed separately for each pair
  dists = [distance(gold, guess)
           for gold, guess
           in zip(gold_data, guess_data)]

  gold_data = list(map(preprocess_for_n_correct, gold_data))
  guess_data = list(map(preprocess_for_n_correct, guess_data))

  # the values needed for P/R can also be broken down per-example
  n_overlaps = [n_correct(gold, guess)
                for gold, guess
                in zip(gold_data, guess_data)]
  gold_lens = [len(gold.split("|")) for gold in gold_data]
  pred_lens = [len(guess.split("|")) for guess in guess_data]

  overall_stats = compute_stats(dists, n_overlaps, gold_lens, pred_lens)
  return overall_stats

METRIC_FUNCTION: MetricFunction = metric_function

compute_metrics(METRIC_FUNCTION, LANGUAGE_CODES)
