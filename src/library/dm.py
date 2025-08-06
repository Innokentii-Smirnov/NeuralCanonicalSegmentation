import os

class DM:

    def __init__(self, direct: str):
        self.direct = direct

    def __enter__(self):
        self.old_direct = os.getcwd()
        os.chdir(self.direct)

    def __exit__(self, exc_type, exc, traceback):
        os.chdir(self.old_direct)
