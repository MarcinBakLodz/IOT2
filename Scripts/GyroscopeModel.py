import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        outputs, _ = self.lstm(x, (hidden, cell))
        outputs = self.output_layer(outputs)
        return outputs


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers)

    def forward(self, x):
        hidden, cell = self.encoder(x)
        # Zakładamy, że decoder dostaje pełną sekwencję zer (lub x przesunięte o 1)
        decoder_input = torch.zeros_like(x)
        output = self.decoder(decoder_input, hidden, cell)
        return output