import json
import numpy as np
from torch import Tensor
from tqdm.auto import tqdm
from collections.abc import Iterable
from utils.vocabulary import Vocabulary
from .randoms import set_random_seed

class Metrics:

    metric_fields = ['sent']

    def __init__(self):
        self.metrics = {"total": 0, "correct": 0, "n_batches": 0, "loss": 0.0}
        self.metrics |= {field+'_corr': 0 for field in self.metric_fields}
        self.metrics |= {field+'_total': 0 for field in self.metric_fields}
        self.metrics |= {field: 0.0 for field in self.metric_fields}

    def __getitem__(self, key: str) -> float | int:
        return self.metrics[key]

    def update(self,
               pred_labels_batch: list[np.ndarray],
               corr_labels_batch: list[np.ndarray],
               loss,
               indexes: list[int],
               dataset,
               val=True):
        
        metrics = self.metrics
        n_batches = metrics["n_batches"]
        metrics["loss"] = (metrics["loss"] * n_batches + loss.item()) / (n_batches + 1)
        metrics["n_batches"] += 1
        batch_size = len(corr_labels_batch)
        # Отдельное предложение
        for corr_labels, pred_labels in zip(corr_labels_batch,
                                            pred_labels_batch,
                                            strict=True):
            # Правильность метки для каждого клитического комплекса
            tokens = np.logical_and(corr_labels == pred_labels, corr_labels != '<UNK>')
            metrics["correct"] += np.sum(tokens)
            metrics["total"] += corr_labels.shape[0]
            metrics["sent_corr"] += np.all(tokens)

        metrics["sent_total"] += batch_size
        metrics["accuracy"] = metrics["correct"] / max(metrics["total"], 1)
        for field in self.metric_fields:
            metrics[field] = metrics[field+'_corr'] / max(metrics[field+"_total"], 1)

    def save(self, filename: str):
        metrics = dict()
        for key, value in self.metrics.items():
            if isinstance(value, np.float64):
                metrics[key] = float(value)
            if isinstance(value, np.int64):
                metrics[key] = int(value)
            else:
                metrics[key] = value
        with open(filename, 'w', encoding='utf-8') as fout1:
            json.dump(metrics, fp=fout1)
    
    @classmethod
    def get_correct_answers(cls, batch):
        raise NotImplementedError
    
    @classmethod
    def convert_to_labels(cls, label_tensors: dict[str, Tensor], vocabs: dict[str, Vocabulary], mask):
        raise NotImplementedError

    @classmethod
    def do_epoch(cls, model, dataloader: Iterable[dict[str, Tensor]], label_vocab: dict[str, Vocabulary], mode="validate", epoch=1):
        val = (epoch == 'evaluate')
        validate = (mode == "validate")
        metrics = cls()
        func = model.train_on_batch if mode == "train" else model.validate_on_batch
        progress_bar = tqdm(dataloader, leave=True)
        progress_bar.set_description(f"{mode}, epoch={epoch}")
        for batch in progress_bar:
            if validate:
                set_random_seed()
            correct_answer = cls.get_correct_answers(batch)
            batch_output = func(batch, correct_answer)
            corr_labels = cls.convert_to_labels(correct_answer, label_vocab, batch["mask"])
            pred_labels = cls.convert_to_labels(batch_output["labels"], label_vocab, batch["mask"])
            metrics.update(pred_labels, corr_labels, batch_output["loss"], batch['indexes'], dataloader.X, val)
            postfix = {"loss": round(metrics["loss"], 4), "acc": round(100 * metrics["accuracy"], 2)}
            postfix |= {field: round(100 * metrics[field], 2) for field in cls.metric_fields}
            progress_bar.set_postfix(postfix)
        return metrics

        
