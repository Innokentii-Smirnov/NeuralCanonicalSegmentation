from metric_computation import compute_metrics, MetricFunction
from morpheme_f1 import compute_f1

LANGUAGE_CODES = ['xhu', 'ces']

METRIC_FUNCTION: MetricFunction = compute_f1

compute_metrics(METRIC_FUNCTION, LANGUAGE_CODES)
