shopt -s extglob
cd models
for lang in *; do
  echo "$lang"
  cd "$lang"
  mkdir tagger
  mv !(tagger) tagger/
  cd ..
done
cd ..
