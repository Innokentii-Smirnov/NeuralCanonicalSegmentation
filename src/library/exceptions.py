class InvalidOperationException(Exception):
    
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

class MissingKeyException(Exception):
    
    def __init__(self, key, dic: dict):
        self.key = key
        self.dic = dic

    def __str__(self):
        string = "The selected index '{0}' is not in the dictionary: {1}".format(self.key, self.dic)
        return string

class InitializationError(Exception):
    
    def __init__(self):
        self.text = 'Initialization failed'

    def __str__(self):
        return self.text

