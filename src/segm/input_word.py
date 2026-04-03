from . import FEATURE_SEP

WORD_FEATURE_SEP = ','

def parse_input_word(input_word: str) -> tuple[str, list[str] | None]:
  match input_word.split(WORD_FEATURE_SEP):
    case word, joined_features:
      features: list[str] | None = joined_features.split(FEATURE_SEP)
    case word,:
      features = None
  return word, features
