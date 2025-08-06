from unicodedata import combining

def string_to_list(string: str) -> list[str]:
    l = list[str]()
    for char in string:
        if combining(char) != 0:
            assert len(l) > 0
            l[-1] += char
        else:
            l.append(char)
    return l

def decode(morphon: list[str], phon: list[str]) -> str:
    for i, (ml, pl) in enumerate(zip(morphon, phon)):
        morphon[i] = ml.replace('C', pl)
    return ''.join(morphon)
    