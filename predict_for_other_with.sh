models="$1"
language_codes="$2"
if [[ -z $language_codes ]] then
  language_codes="chu vsn"
fi
function predict_for_language() {
  code="$1"
  infile="data/$code.word.test.tsv"
  for model in $models; do
    for subtype in $(ls "models/$code/$model"); do
      outdir="predictions/$code/$model/$subtype"
      mkdir -p "$outdir"
      outfile="$outdir/predictions.txt"
      cat "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" | tr -d '()' > "$outfile"
    done
  done
}
for code in $language_codes; do
  predict_for_language "$code"
done
