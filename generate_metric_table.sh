metric="$1"
env/bin/python src/generate_metric_table.py "metrics/$metric.json" \
  --languages chu eng deu ind xhu \
  --models tagger_{CNN,LSTM,RCNN,RCNN-skip-conn} \
           hard-attention_CLUZH \
           transducer_LSTM \
           transformer_{char,char_bs128} \
  | xclip -selection clipboard
xclip -selection clipboard -o
