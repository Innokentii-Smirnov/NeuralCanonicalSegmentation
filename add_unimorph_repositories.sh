language_codes="$1"
for lang in $language_codes; do git submodule add "https://github.com/unimorph/$lang.git" "unimorph/$lang"; done
