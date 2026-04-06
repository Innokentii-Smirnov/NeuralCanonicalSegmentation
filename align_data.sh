language="$1"
code="$2"
aligned_data_dir="aligned_data/Levenshtein/$language"
if [[ -d "$aligned_data_dir" ]] then
  rm -r "$aligned_data_dir"
fi
dotnet run --project LevenshteinAlignment "$language" "$aligned_data_dir" alignment_costs/$code
