indir="canonical-segmentation"
outdir="reformatted_canonical-segmentation"
function reformat_for_language() {
  language="$1"
  code="$2"
  for split in 0 1 2 3 4 5 6 7 8 9; do
    insubdir="$indir/$language"
    outsubdir="$outdir/$split"
    mkdir -p "$outsubdir"
    for part in train dev; do
      cut -f1,3 "$insubdir/$part$split" > "$outsubdir/$code.word.$part.tsv"
    done
    cut -f1,3 "$insubdir/test0" > "$outsubdir/$code.word.test.gold.tsv"
    cut -f1 "$insubdir/test0" > "$outsubdir/$code.word.test.tsv"
  done
}
reformat_for_language english eng
reformat_for_language german deu
reformat_for_language indonesian ind
