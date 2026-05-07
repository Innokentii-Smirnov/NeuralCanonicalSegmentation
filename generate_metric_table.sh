metric="$1"
language_codes="$2"
if [ -z "$language_codes" ]; then
  language_codes="chu eng deu ind xhu"
fi
env/bin/python src/generate_metric_table.py "metrics/$metric.json" \
  --languages $language_codes \
  --models tagger_{CNN,LSTM,RCNN,RCNN-skip-conn} \
           hard-attention_CLUZH \
           transducer_LSTM \
           transformer_{char,char_bs128} \
  | xclip -selection clipboard
xclip -selection clipboard -o
