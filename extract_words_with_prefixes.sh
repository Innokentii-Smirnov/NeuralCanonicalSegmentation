lang="$1"
prefixes="$2"
for prefix in $prefixes; do cut -f 1 "unimorph/$lang/$lang" | awk '{print tolower($0)}' | grep "^$prefix" | sort | uniq > "examples/$lang/$prefix.txt"; done
