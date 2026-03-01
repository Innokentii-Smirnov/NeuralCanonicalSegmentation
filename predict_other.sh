function predict_for_language() {
  code="$1"
  infile="data/$code.word.test.tsv"
  for model in $(ls "models/$code"); do
    for subtype in $(ls "models/$code/$model"); do
      outdir="predictions/$code/$model/$subtype"
      mkdir -p "$outdir"
      outfile="$outdir/predictions.txt"
      cat "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" > "$outfile"
    done
  done
}
for code in chu; do
  predict_for_language "$code"
done
