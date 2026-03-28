import torch
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from stringutils import string_to_list, decode
from models.transformer.transformer import Seq2SeqTransformer

def get_word(li: list[str]) -> str:
    if '<BEGIN>' in li and '<END>' in li:
        start = li.index('<BEGIN>')
        end = li.index('<END>')
        if end > start:
            return ''.join(li[start+1:end])
    return ''.join(li)

class MorphonologicalTransformerApplier:

    def __init__(self,
                 model: Seq2SeqTransformer,
                 vocabularies: dict[str, Vocabulary], device: torch.device):
        self.model = model
        self.device = device
        if self.device is not None:
            self.model.to(self.device)
        self.vocabularies = vocabularies
        self.combine_diacritics = self.vocabularies['phon'].contains_string_with_combined_diacritic()

    def predict(self, X: SimpleDataset, batch_size: int = 32, max_len: int = 50) -> list[None | ndarray]:
        self.model.eval()
        dataloader = FieldBatchDataloader(X, device=self.device, batch_size=batch_size)
        answer: list[None | ndarray] = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            num_tokens = batch['phon'].shape[1]
            src = batch['phon'].transpose(0, 1)
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self.device)
            with torch.no_grad():
                tgt_tokens = self.model.greedy_decode(src, src_mask, max_len=max_len, start_symbol=self.model.begin_token_id)
            labels = tgt_tokens.transpose(1, 0).cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for index, curr_labels in zip(indexes, labels, strict=True):
                result = np.take(X.vocabs["morphon"].symbols_, curr_labels)
                answer[index] = result
        return answer

    def translate(self, src_sentence: str, max_len: int = 50):
        self.model.eval()
        inp = string_to_list(src_sentence, self.combine_diacritics)
        src = torch.LongTensor(self.model.src_vocab.vectorize_element(inp)).to(self.device).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self.device)
        tgt_tokens = self.model.greedy_decode(src, src_mask, max_len=max_len, start_symbol=self.model.begin_token_id).flatten()
        curr_labels = tgt_tokens.cpu().numpy()
        return np.take(self.model.tgt_vocab.symbols_, curr_labels)

    def apply_to(self, words: list[str]) -> list[str]:
        predictions = list[str]()
        for word in tqdm(words):
            prediction = self.translate(word)
            morphon = get_word(list(prediction))
            predictions.append(morphon)
        return predictions
