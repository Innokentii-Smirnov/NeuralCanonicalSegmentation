data_dir="$1"
rm -r aligned_data/Levenshtein/Hurrian
dotnet run --project LevenshteinAlignment "$data_dir"/xhu.word.{train,dev}.tsv aligned_data/Levenshtein/Hurrian alignment_costs/xhu
