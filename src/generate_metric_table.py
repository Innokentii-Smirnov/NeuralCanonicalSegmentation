from argparse import ArgumentParser
import json
from collections import defaultdict
import pandas as pd

parser = ArgumentParser(
  prog='generate_metric_table.py',
  description='Generate a table with columns for models, rows for languages and metric values in the cells'
)
parser.add_argument('infile',
                    help='A JSON file containing a dictionary of the form {language: {model: metric_value}}')
parser.add_argument('--languages', nargs='*',
                    help='ISO-639-3 codes of the languages to include in the table')
parser.add_argument('--models', nargs='*',
                    help='the names of the models to include in the table')
args = parser.parse_args()

with open(args.infile, 'r', encoding='utf-8') as fin:
  data = json.load(fin)
metric_values = defaultdict(dict)
for model in args.models:
  model_name = model.split('_')[1]
  key = model_name if model_name not in metric_values else model_name + '2'
  for language in args.languages:
    metric_values[key][language] = data[language].get(model, 0.0)
df = pd.DataFrame(metric_values)
print(df.to_latex(float_format="%.2f"))
