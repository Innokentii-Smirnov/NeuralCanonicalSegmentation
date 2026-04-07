import numpy as np
import torch.nn as nn
from torch import Tensor

from tqdm.auto import tqdm
from typing import Iterable

from metrics import Metrics
from utils.vocabulary import Vocabulary
from trainers import Trainer
from trainers.sequence_generation import SequenceGeneratorTrainer

def convert_to_labels(label_tensor: Tensor, vocab: Vocabulary, mask):
    label_array = label_tensor.detach().cpu().numpy()
    if mask is None:
      labels = [np.take(vocab.symbols_, word) for word in label_array]
    else:
      mask = mask.bool().detach().cpu().numpy()
      labels = [np.take(vocab.symbols_, word[curr_mask]) for word, curr_mask in zip(label_array, mask)]
    return labels

def do_epoch(trainer: Trainer, dataloader: Iterable[dict[str, Tensor]], label_vocab: Vocabulary, mode="validate", epoch=1):
    metrics = Metrics()
    func = trainer.train_on_batch if mode == "train" else trainer.validate_on_batch
    progress_bar = tqdm(dataloader, leave=True)
    progress_bar.set_description(f"{mode}, epoch={epoch}")

    for batch in progress_bar:
        batch_output, y = func(batch, batch["morphon"])
        mask = None if isinstance(trainer, SequenceGeneratorTrainer) else batch["mask"]
        corr_labels = convert_to_labels(y, label_vocab, mask)
        pred_labels = convert_to_labels(batch_output["labels"], label_vocab, mask)
        if isinstance(trainer, SequenceGeneratorTrainer):
            pred_labels = [pred_letters[:-1] for pred_letters in pred_labels]
            corr_labels = [corr_letters[1:] for corr_letters in corr_labels]
        metrics.update(pred_labels, corr_labels, batch_output["loss"])
        postfix = {"loss": round(metrics.loss, 4), "acc": round(100 * metrics.accuracy, 2)}
        progress_bar.set_postfix(postfix)
    return metrics
