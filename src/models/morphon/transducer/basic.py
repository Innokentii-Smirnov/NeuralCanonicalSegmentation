import torch
from basic_models.sequence_generation import BasicSequenceGenerator
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from stringutils import string_to_list

class BasicMorphonologicalTransducer(BasicSequenceGenerator):

    def __init__(self, vocabularies: dict[str, Vocabulary], device: torch.device):
        super(BasicMorphonologicalTransducer, self).__init__(device)
        self.vocabularies = vocabularies
        self.combine_diacritics = self.vocabularies['phon'].contains_string_with_combined_diacritic()

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
