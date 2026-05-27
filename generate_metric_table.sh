metric="$1"
ndigits="$2"
language_codes="$3"
if [ -z "$language_codes" ]; then
  language_codes="chu eng deu ind xhu"
fi
if [ -z "$ndigits" ]; then
  ndigits="2"
fi
env/bin/python src/generate_metric_table.py "metrics/$metric.json" \
  --languages $language_codes \
  --models tagger_{CNN,LSTM,RCNN,RCNN-skip-conn} \
           hard-attention_CLUZH \
           transducer_LSTM \
           transformer_{char,char_bs128} \
  --ndigits "$ndigits" \
  | xclip -selection clipboard
xclip -selection clipboard -o
