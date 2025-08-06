def in_braces(s: str) -> bool:
    return s.startswith('{') and s.endswith('}')

def in_angles(s: str) -> bool:
    return s.startswith('<') and s.endswith('>')

def debrace(s: str):
    assert s.startswith('{') and s.endswith('}')
    ans = s[1:-1]
    return ans

def find_with_brack_balance(string: str, symbol: str,
                                 opening: str, closing: str):
    results = []
    balance = 0
    for i, x in enumerate(string):
        if x == opening:
            balance += 1
        elif x == closing:
            balance -= 1
        elif x == symbol:
            if balance == 0:
                results.append(i)
    return results

def enquote(x: str):
    return "'"+x+"'"
