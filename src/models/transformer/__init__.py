import torch
from utils.vocabulary import Vocabulary
from .transformer import Seq2SeqTransformer

EMB_SIZE = 256
NHEAD = 8
FFN_HID_DIM = 4 * EMB_SIZE
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

def make_model(vocabularies: dict[str, Vocabulary], device: torch.device) -> Seq2SeqTransformer:
    src_vocab = vocabularies['phon']
    tgt_vocab = vocabularies['morphon']
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                              NHEAD, src_vocab, tgt_vocab, device,
                              FFN_HID_DIM).to(device)
    return model
