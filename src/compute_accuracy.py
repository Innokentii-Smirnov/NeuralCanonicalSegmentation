from sklearn.metrics import accuracy_score
from metric_computation import compute_metrics, MetricFunction

LANGUAGE_CODES = ['deu', 'eng', 'ind', 'ita', 'fra', 'lat', 'chu', 'vsn', 'xhu', 'ces']

METRIC_FUNCTION: MetricFunction = lambda y_true, y_pred: \
  {'accuracy': float(100 * accuracy_score(y_true, y_pred))}

compute_metrics(METRIC_FUNCTION, LANGUAGE_CODES)
