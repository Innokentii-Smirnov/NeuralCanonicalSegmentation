models="$1"
language_codes="$2"
dataset="$3"
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
    for subtype in $(ls "models/$code/$model"); do
      outdir="predictions/$code/$model/$subtype"
      mkdir -p "$outdir"
      outfile="$outdir/predictions.txt"
      if [[ "$subtype" == *-pos ]] then
        cut -f1,2 --output-delimiter=',' "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" | tr -d '()' > "$outfile"
      else
        cut -f1 "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" | tr -d '()' > "$outfile"
      fi
    done
  done
}
for code in $language_codes; do
  predict_for_language "$code"
done
