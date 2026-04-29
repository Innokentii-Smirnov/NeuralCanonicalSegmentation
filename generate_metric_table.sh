env/bin/python src/generate_metric_table.py metrics/Accuracy.json \
  --languages chu eng deu ind \
  --models tagger_{CNN,LSTM,RCNN,RCNN-skip-conn} \
           hard-attention_CLUZH \
           transducer_LSTM \
           transformer_char \
  | xclip -selection clipboard
xclip -selection clipboard -o
