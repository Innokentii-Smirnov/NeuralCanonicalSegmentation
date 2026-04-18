dataset="$1"
language_code="$2"
epochs="$3"
model_directory="$4"
python src/train.py "$dataset" "$language_code" '~' transformer char "$epochs" \
  "$model_directory" --num-encoder-layers 4 --num-decoder-layers 4 --emb-size 256 \
  --nhead 4 --dim-feedforward 1024 --dropout 0.3 --batch-size 400
