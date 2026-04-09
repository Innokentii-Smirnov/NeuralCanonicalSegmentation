language="$1"
code="$2"
mkdir -p "$language"
for partition in train dev test.gold; do
  sed 's/ @@/-/g' "2022SegmentationST/data/$code.word.$partition.tsv" > "$language/$code.word.$partition.tsv"
done
cp "2022SegmentationST/data/$code.word.test.tsv" "$language/$code.word.test.tsv"
