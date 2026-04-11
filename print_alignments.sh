language="$1"
code="$2"
aligned_data_dir="aligned_data/Levenshtein/$language"
if [[ ! -d "$aligned_data_dir" ]] then
  echo "The directory $aligned_data_dir does not exist."
fi
env/bin/python src/print_alignments.py $aligned_data_dir/$code.word.train.tsv formatted_alignments/Levenshtein/$code
env/bin/python src/print_alignments.py $aligned_data_dir/$code.word.dev.tsv formatted_alignments/Levenshtein/$code-dev
