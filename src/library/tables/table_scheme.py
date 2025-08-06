from typing import Callable
import numpy as np
from numpy import ndarray
from .table import Table
from . import linearize

class TableScheme:

    def __init__(self, row_cats: list[str], col_cats: list[str],
                 row_feats: list[list[str]], col_feats: list[list[str]]):
        self.row_cats = np.array(row_cats)
        self.col_cats = np.array(col_cats)
        self.row_feats = np.array(row_feats)
        self.col_feats = np.array(col_feats)

    def get_contents(self, parfunc: Callable[[dict[str, str]], str] | None = None) -> ndarray:
        func = parfunc or linearize
        contents = np.ndarray((self.row_feats.shape[0], self.col_feats.shape[0]), dtype=object)
        row_gramm_form = dict[str, str]()
        for i in range(contents.shape[0]):
            row_feats = self.row_feats[i]
            for row_cat, row_feat in zip(self.row_cats, row_feats, strict=True):
                row_gramm_form[row_cat] = row_feat
            for j in range(contents.shape[1]):
                gramm_form = row_gramm_form.copy()
                col_feats = self.col_feats[j]
                for col_cat, col_feat in zip(self.col_cats, col_feats, strict=True):
                    gramm_form[col_cat] = col_feat
                contents[i, j] = func(gramm_form)
        return contents

    def get_table(self, parfunc: Callable[[dict[str, str]], str] | None = None) -> Table:
        return Table(self, self.get_contents(parfunc))
