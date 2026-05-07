from argparse import ArgumentParser
from metric_computation import compute_metrics, MetricFunction
from morpheme_f1 import compute_f1

LANGUAGE_CODES = ['chu', 'eng', 'deu', 'ind', 'xhu', 'ces']

parser = ArgumentParser(
  prog='compute_f1.py',
  description='Compute F1-measure for canonical morpheme segmentation'
)
parser.add_argument('--language-codes', nargs='*', default=LANGUAGE_CODES)
args = parser.parse_args()

METRIC_FUNCTION: MetricFunction = compute_f1

compute_metrics(METRIC_FUNCTION, args.language_codes)
