model_types="$1"
function predict_for_language() {
  code="$1"
  indir="examples/$code"
  for file_name in $(ls "$indir"); do
    infile="$indir/$file_name"
    for model in $model_types; do
      for subtype in $(ls "models/$code/$model"); do
        outdir="segmented_examples/$lang/$model/$subtype"
        mkdir -p "$outdir"
        outfile="$outdir/$file_name"
        cat "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" | tr -d '()' > "$outfile"
      done
    done
  done
}
for code in $(ls examples); do
  predict_for_language "$code" "$model_types"
done
