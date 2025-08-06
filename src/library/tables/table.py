import numpy as np
from numpy import ndarray
from ..align import align_inplace

class Table:
    
    def __init__(self, scheme, contents: ndarray):
        self.scheme = scheme
        self.contents = contents
        self.space = 2

    def get_array(self) -> ndarray:
        scheme = self.scheme
        shape = (scheme.col_cats.shape[0] + scheme.row_feats.shape[0], scheme.row_cats.shape[0] + scheme.col_feats.shape[0])
        array = np.ndarray(shape, dtype=object)

        array[scheme.col_cats.shape[0] - 1, 0 : scheme.row_cats.shape[0] - 1] = scheme.row_cats[:-1]
        array[0 : scheme.col_cats.shape[0] - 1, scheme.row_cats.shape[0] - 1] = scheme.col_cats[:-1]
        array[scheme.col_cats.shape[0] - 1, scheme.row_cats.shape[0] - 1] = '/'.join([scheme.row_cats[-1], scheme.col_cats[-1]])

        array[scheme.col_cats.shape[0]:, 0 : scheme.row_cats.shape[0]] = scheme.row_feats
        array[0 : scheme.col_cats.shape[0], scheme.row_cats.shape[0]:] = scheme.col_feats.transpose()
        array[scheme.col_cats.shape[0]:, scheme.row_cats.shape[0]:] = self.contents
        return array

    def __repr__(self) -> str:
        array = self.get_array()
        for ncol in range(array.shape[1]):
            align_inplace(array[:, ncol])
        rows = ((' ' * self.space).join(array[i]) for i in range(array.shape[0]))
        string = '\n'.join(rows)
        return string
    
    def to_latex(self) -> str:
        array = self.get_array()
        for ncol in range(array.shape[1]):
            align_inplace(array[:, ncol])
        rows = (' & '.join(array[i]) for i in range(array.shape[0]))
        string = ' \\\\\n'.join(rows)
        return string
