from utils.vocabulary import Vocabulary

class NetworkArguments:

    def __init__(self, embedding_dim: int, hidden_size: int,
                 num_layers: int, lstm_dropout: float):

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_dropout = lstm_dropout

class EncoderArguments(NetworkArguments):

    def __init__(self, vocab_size: int, embedding_dim: int, embedding_dropout: float,
                 hidden_size: int, num_layers: int, lstm_dropout: float, bidirectional: bool):

        self.vocab_size = vocab_size
        self.embedding_dropout = embedding_dropout
        super().__init__(embedding_dim, hidden_size, num_layers, lstm_dropout)
        self.bidirectional = bidirectional

class DecoderArguments(NetworkArguments):

    def __init__(self, vocabulary: Vocabulary, embedding_dim: int,
                 hidden_size: int, num_layers: int, lstm_dropout: float):

        self.vocabulary = vocabulary
        super().__init__(embedding_dim, hidden_size, num_layers, lstm_dropout)

class CNNArguments:

    def __init__(self, input_dim: int,
                 n_layers: int,
                 window: int | list[int],
                 n_hidden: int | list[int],
                 dropout: float,
                 use_batch_norm: bool):
        
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.window = window
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
