language="$1"
code="$2"
clear && ./align_data.sh "$language" "$code" && ./print_alignments.sh "$language" "$code"
