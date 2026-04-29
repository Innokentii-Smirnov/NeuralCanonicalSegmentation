input_directory="$1"
dataset="$2"
code="$3"
clear && ./align_data.sh "$input_directory" "$dataset" "$code" && ./print_alignments.sh "$dataset" "$code"
