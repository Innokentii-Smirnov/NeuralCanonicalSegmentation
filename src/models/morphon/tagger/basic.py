import torch
from basic_models.mc import BasicMc
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from stringutils import string_to_list, decode

class BasicMorphonologicalTransducer(BasicMc):
    
    def __init__(self, vocabularies: dict[str, Vocabulary], device: torch.device):
        super(BasicMorphonologicalTransducer, self).__init__(device)
        self.vocabularies = vocabularies

    def apply(self, words: list[str], decode_copy: bool = False) -> list[str]:
        data = [{'phon': string_to_list(word)} for word in words]
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
