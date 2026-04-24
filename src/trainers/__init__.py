import torch
from .mc import McTrainer
from .sequence_generation import SequenceGeneratorTrainer
from .transformer import TransformerTrainer

Trainer = McTrainer | SequenceGeneratorTrainer | TransformerTrainer

def make_trainer(model_type: str, model, device: torch.device, args):
  match model_type:
    case 'tagger':
      return McTrainer(model, device)
    case 'transducer':
      return SequenceGeneratorTrainer(model, device, args.scheduling)
    case 'transformer':
      return TransformerTrainer(model, device)
    case _:
      raise ValueError('Unsupported model type: ' + model_type)
