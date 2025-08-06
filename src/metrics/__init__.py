import numpy as np
from numpy import ndarray

class Metrics:

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.n_batches = 0
        self.loss = 0.0

    def update(self, pred_letters_batch: list[ndarray],
               corr_letters_batch: list[ndarray], loss):

        self.loss = (self.loss * self.n_batches + loss.item()) / (self.n_batches + 1)
        self.n_batches += 1

        for corr_letters, pred_letters in zip(corr_letters_batch,
                                              pred_letters_batch,
                                              strict=True):

            correct_letters = (pred_letters == corr_letters)
            correct_token = np.all(correct_letters)
            self.correct += int(correct_token)
            self.total += 1

        self.accuracy = self.correct / max(self.total, 1)

    def __repr__(self) -> str:
        return 'Accuracy = {0} %  ({1}/{2})'.format(round(100*self.accuracy, 2), self.correct, self.total, )
