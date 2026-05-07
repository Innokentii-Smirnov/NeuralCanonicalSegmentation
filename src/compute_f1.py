from argparse import ArgumentParser
from functools import partial
from metric_computation import compute_metrics, MetricFunction
from morpheme_f1 import compute_f1

LANGUAGE_CODES = ['chu', 'eng', 'deu', 'ind', 'xhu', 'ces']

parser = ArgumentParser(
  prog='compute_f1.py',
  description='Compute F1-measure for canonical morpheme segmentation'
)
parser.add_argument('--language-codes', nargs='*', default=LANGUAGE_CODES)
parser.add_argument('--ignore-boundary-type', action='store_true')
args = parser.parse_args()

METRIC_FUNCTION: MetricFunction = partial(compute_f1, ignore_boundary_type=args.ignore_boundary_type)

compute_metrics(METRIC_FUNCTION, args.language_codes)
