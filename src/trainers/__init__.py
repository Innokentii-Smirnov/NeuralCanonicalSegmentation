import torch
from .mc import McTrainer
from .sequence_generation import SequenceGeneratorTrainer

Trainer = McTrainer | SequenceGeneratorTrainer

def make_trainer(model_type: str, model, device: torch.device):
  match model_type:
    case 'tagger':
      return McTrainer(model, device)
    case 'transducer':
      return SequenceGeneratorTrainer(model, device)
    case 'transformer':
      raise NotImplementedError
    case _:
      raise ValueError('Unsupported model type: ' + model_type)
