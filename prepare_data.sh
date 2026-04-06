language="$1"
code="$2"
lowercased_language=$(echo "$language" | tr '[:upper:]' '[:lower:]')
echo "$lowercased_language"
mkdir -p "$language"
for partition in train dev; do
  cut -f1,3 canonical-segmentation/$lowercased_language/${partition}0 | sed 's/ /@/g' > "$language/$code.word.$partition.tsv"
done
cut -f1,3 canonical-segmentation/$lowercased_language/test0 | sed 's/ /@/g' > "$language/$code.word.test.gold.tsv"
cut -f1 canonical-segmentation/$lowercased_language/test0 > "$language/$code.word.test.tsv"
