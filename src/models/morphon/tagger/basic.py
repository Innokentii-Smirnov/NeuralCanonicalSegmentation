import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from stringutils import string_to_list, decode

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
                batch_answer = self(batch['phon'])
            labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for index, curr_labels, curr_mask in zip(indexes, labels, batch['mask'].bool().cpu().numpy(), strict=True):
                result = np.take(X.vocabs["morphon"].symbols_, curr_labels[curr_mask])
                answer[index] = result
        return answer

    def apply_to(self, words: list[str], decode_copy: bool = False) -> list[str]:
        data = [{'phon': string_to_list(word, self.combine_diacritics)} for word in words]
        dataset = SimpleDataset(data, ['phon'], [],
            True, True, True, True, self.vocabularies
        )
        predictions = self.predict(dataset)
        segmentations = list[str]()
        if decode_copy:
            for i, prediction in enumerate(predictions):
                segmentation = decode(prediction, data[i]['phon'])
                segmentations.append(segmentation)
        else:
            for prediction in predictions:
                segmentations.append(''.join(prediction))
        return segmentations
