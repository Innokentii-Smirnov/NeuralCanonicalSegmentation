from unicodedata import combining

def is_combining_diacritic(character: str) -> bool:
    return combining(character) != 0

def contains_combined_diacritic(string: str) -> bool:
    if len(string) < 2:
        return False
    for character in string[1:]:
        if is_combining_diacritic(character):
            return True
    return False

def string_to_list(string: str, combine_diacritics: bool = False) -> list[str]:
    if combine_diacritics:
        return string_to_list_with_combined_diacritics(string)
    else:
        return list(string)

def string_to_list_with_combined_diacritics(string: str) -> list[str]:
    l = list[str]()
    for char in string:
        if is_combining_diacritic(char):
            assert len(l) > 0
            l[-1] += char
        else:
            l.append(char)
    return l

def decode(morphon: list[str], phon: list[str]) -> str:
    for i, (ml, pl) in enumerate(zip(morphon, phon)):
        morphon[i] = ml.replace('C', pl)
    return ''.join(morphon)
    
