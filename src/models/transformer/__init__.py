import torch
from utils.vocabulary import Vocabulary
from .transformer import Seq2SeqTransformer
from arguments.transformer import Hyperparameters

def make_model(vocabularies: dict[str, Vocabulary], hyperparameters: Hyperparameters, device: torch.device) -> Seq2SeqTransformer:
    src_vocab = vocabularies['phon']
    tgt_vocab = vocabularies['morphon']
    model = Seq2SeqTransformer(src_vocab, tgt_vocab, **hyperparameters, device=device).to(device)
    return model
