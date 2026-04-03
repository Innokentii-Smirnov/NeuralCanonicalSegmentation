models="$1"
language_codes="$2"
dataset="$3"
model_names="$4"
if [[ -z $language_codes ]] then
  language_codes="chu vsn"
fi
if [[ -z $dataset ]] then
  dataset="data"
fi
function predict_for_language() {
  code="$1"
  infile="$dataset/$code.word.test.tsv"
  for model in $models; do
    if [[ -z $model_names ]] then
      model_variants=$(ls "models/$code/$model")
    else
      model_variants="$model_names"
    fi
    for model_name in $model_variants; do
      outdir="predictions/$code/$model/$model_name"
      mkdir -p "$outdir"
      outfile="$outdir/predictions.txt"
      subtype="${model_name%%_*}"
      model_directory="models/$code/$model/$model_name"
      if [[ "$subtype" == *-pos ]] then
        cut -f1,2 --output-delimiter=',' "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" --model-directory "$model_directory" | tr -d '()' > "$outfile"
      else
        cut -f1 "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" --model-directory "$model_directory" | tr -d '()' > "$outfile"
      fi
    done
  done
}
for code in $language_codes; do
  predict_for_language "$code"
done
