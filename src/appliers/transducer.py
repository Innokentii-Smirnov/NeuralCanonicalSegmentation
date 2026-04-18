import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from stringutils import string_to_list
from transducers.rnn.sequence import SequenceTransducer

class MorphonologicalTransducerApplier:

    def __init__(self, model: SequenceTransducer, vocabularies: dict[str, Vocabulary], device: torch.device,
                 max_output_length: int = 50):
        self.model = model
        self.device = device
        if self.device is not None:
            self.model.to(self.device)
        self.vocabularies = vocabularies
        self.combine_diacritics = self.vocabularies['phon'].contains_string_with_combined_diacritic()
        self.max_output_length = max_output_length

    def predict(self, X: SimpleDataset, batch_size: int) -> list[ndarray]:
        self.model.eval()
        dataloader = FieldBatchDataloader(X, device=self.device, batch_size=batch_size)
        answer: list[ndarray] = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            with torch.no_grad():
                batch_answer = self.model.transduce(batch['phon'], self.max_output_length, features=batch.get('features', None))
            labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for index, curr_labels in zip(indexes, labels, strict=True):
                result = np.take(X.vocabs["morphon"].symbols_, curr_labels)
                answer[index] = result
        return answer

    def apply_to(self, words: list[tuple[str, list[str] | None]], batch_size: int = 32) -> list[str]:
        data = [{'phon': string_to_list(word, self.combine_diacritics), 'features': features} for word, features in words]
        dataset = SimpleDataset(data, ['phon', 'features'], [],
            True, True, True, True, self.vocabularies
        )
        predictions = self.predict(dataset, batch_size)
        segmentations = list[str]()
        for prediction in predictions:
            segmentation = ''.join(letter for letter in prediction if letter not in {'<PAD>', '<BEGIN>', '<END>'})
            segmentations.append(segmentation)
        return segmentations
