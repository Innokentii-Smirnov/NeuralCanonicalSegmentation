input_directory="$1"
dataset="$2"
code="$3"
aligned_data_dir="aligned_data/Levenshtein/$dataset"
if [[ -d "$aligned_data_dir" ]] then
  rm "$aligned_data_dir"/$code.word.{train,dev}.tsv
fi
dotnet run --project LevenshteinAlignment $input_directory/$code.word.{train,dev}.tsv "$aligned_data_dir" alignment_costs/$code
