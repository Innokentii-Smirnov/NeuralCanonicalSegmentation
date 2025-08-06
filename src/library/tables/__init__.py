import re

def linearize(gramm_form: dict[str, str]):
    return re.sub(r'(?<=\d)\.', '', '.'.join(filter(lambda x: not x.startswith('_'), gramm_form.values())).upper())
