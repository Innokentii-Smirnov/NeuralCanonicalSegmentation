import numpy as np
import torch.nn as nn
from torch import Tensor

from tqdm.auto import tqdm
from typing import Iterable

from metrics import Metrics
from utils.vocabulary import Vocabulary
from basic_models.mc import BasicMc

def convert_to_labels(label_tensor: Tensor, vocab: Vocabulary, mask):
    mask = mask.bool().detach().cpu().numpy()
    label_array = label_tensor.detach().cpu().numpy()
    labels = [np.take(vocab.symbols_, word[curr_mask]) for word, curr_mask in zip(label_array, mask)]
    return labels

def do_epoch(model: BasicMc, dataloader: Iterable[dict[str, Tensor]], label_vocab: Vocabulary, mode="validate", epoch=1):
    metrics = Metrics()
    func = model.train_on_batch if mode == "train" else model.validate_on_batch
    progress_bar = tqdm(dataloader, leave=True)
    progress_bar.set_description(f"{mode}, epoch={epoch}")
    if mode == 'validate':
        generate = True
    elif mode == 'train':
        generate = False
    else:
        raise ValueError(mode)

    for batch in progress_bar:
        batch_output = func(batch, batch["morphon"])
        corr_labels = convert_to_labels(batch['morphon'], label_vocab, batch["mask"])
        pred_labels = convert_to_labels(batch_output["labels"], label_vocab, batch["mask"])
        metrics.update(pred_labels, corr_labels, batch_output["loss"])
        postfix = {"loss": round(metrics.loss, 4), "acc": round(100 * metrics.accuracy, 2)}
        progress_bar.set_postfix(postfix)
    return metrics
