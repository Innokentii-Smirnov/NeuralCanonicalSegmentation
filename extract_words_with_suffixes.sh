lang="$1"
suffixes="$2"
for suffix in $suffixes; do cut -f 1 "unimorph/$lang/$lang" | awk '{print tolower($0)}' | grep "$suffix\$" | sort | uniq > "examples/$lang/$suffix.txt"; done
