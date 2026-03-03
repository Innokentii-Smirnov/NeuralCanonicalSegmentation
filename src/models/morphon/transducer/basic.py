import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from stringutils import string_to_list

class BasicMorphonologicalTransducer(nn.Module):

    def __init__(self, vocabularies: dict[str, Vocabulary], device: torch.device):
        super(BasicMorphonologicalTransducer, self).__init__()
        self.device = device
        if self.device is not None:
            self.to(self.device)
        self.vocabularies = vocabularies
        self.combine_diacritics = self.vocabularies['phon'].contains_string_with_combined_diacritic()

    def predict(self, X: SimpleDataset) -> list[ndarray]:
        self.eval()
        dataloader = FieldBatchDataloader(X, device=self.device, batch_size=32)
        answer: list[ndarray] = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            with torch.no_grad():
                batch_answer = self(batch['phon'], None, True)
            labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for index, curr_labels in zip(indexes, labels, strict=True):
                result = np.take(X.vocabs["morphon"].symbols_, curr_labels)
                answer[index] = result
        return answer

    def apply_to(self, words: list[str]) -> list[str]:
        data = [{'phon': string_to_list(word, self.combine_diacritics)} for word in words]
        dataset = SimpleDataset(data, ['phon'], [],
            True, True, True, True, self.vocabularies
        )
        predictions = self.predict(dataset)
        segmentations = list[str]()
        for prediction in predictions:
            segmentation = ''.join(letter for letter in prediction if letter not in {'<PAD>', '<BEGIN>', '<END>'})
            segmentations.append(segmentation)
        return segmentations
