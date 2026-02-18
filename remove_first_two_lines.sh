for code in $(ls predictions); do
  for model in $(ls "predictions/$code"); do
    for subtype in $(ls "predictions/$code/$model"); do
      dir="predictions/$code/$model/$subtype"
      file="$dir/predictions.txt"
      sed -e '1d;2d' -i "$file"
    done
  done
done
