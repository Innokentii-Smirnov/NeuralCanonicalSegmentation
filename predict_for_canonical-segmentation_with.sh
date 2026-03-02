models="$1"
function predict_for_language() {
  code="$1"
  language="$2"
  infile="canonical-segmentation/$language/test0"
  for model in $models; do
    for subtype in $(ls "models/$code/$model"); do
      outdir="predictions/$code/$model/$subtype"
      mkdir -p "$outdir"
      outfile="$outdir/predictions.txt"
      cut -f 1 "$infile" | env/bin/python src/segment.py "$code" "$model" "$subtype" > "$outfile"
    done
  done
}
predict_for_language eng english
predict_for_language deu german
predict_for_language ind indonesian
