
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ELMoCharacterEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim=16, char_cnn_filters=[[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]):
        super().__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        
        # Character CNN layers
        self.convolutions = nn.ModuleList([
            nn.Conv1d(char_embed_dim, num_filters, kernel_size)
            for kernel_size, num_filters in char_cnn_filters
        ])
        
        self.output_dim = sum(f[1] for f in char_cnn_filters)
        self.highways = Highway(self.output_dim, num_layers=2)

    def forward(self, chars):
        # chars: [batch_size, seq_len, word_len]
        batch_size, seq_len, word_len = chars.size()
        chars = chars.view(-1, word_len)  # [batch_size * seq_len, word_len]
        
        char_embeds = self.char_embed(chars)  # [batch_size * seq_len, word_len, char_embed_dim]
        char_embeds = char_embeds.transpose(1, 2)  # [batch_size * seq_len, char_embed_dim, word_len]
        
        conv_outputs = []
        for conv in self.convolutions:
            conv_output = conv(char_embeds)  # [batch_size * seq_len, num_filters, word_len - kernel_size + 1]
            conv_output = torch.max(conv_output, dim=-1)[0]  # [batch_size * seq_len, num_filters]
            conv_outputs.append(conv_output)
        
        char_embeddings = torch.cat(conv_outputs, dim=-1)  # [batch_size * seq_len, total_filters]
        char_embeddings = self.highways(char_embeddings)
        
        return char_embeddings.view(batch_size, seq_len, -1)  # [batch_size, seq_len, total_filters]

class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            projected = layer(x)
            transform_gate = torch.sigmoid(projected[..., :self.input_dim])
            carry_gate = 1 - transform_gate
            nonlinear = torch.relu(projected[..., self.input_dim:])
            x = transform_gate * nonlinear + carry_gate * x
        return x

class ELMo(nn.Module):
    def __init__(self, char_vocab_size, hidden_size=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.char_encoder = ELMoCharacterEncoder(char_vocab_size)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.char_encoder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projections for forward and backward language models
        self.forward_proj = nn.Linear(hidden_size, char_vocab_size)
        self.backward_proj = nn.Linear(hidden_size, char_vocab_size)
        
        # Scalar parameters for computing weighted sum of layers
        self.scalar_parameters = nn.Parameter(torch.zeros(num_layers * 2 + 1))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, chars, lengths):
        # Get character-level embeddings
        char_embeddings = self.char_encoder(chars)  # [batch_size, seq_len, char_encoder_dim]
        
        # Pack padded sequence for LSTM
        packed_embeddings = pack_padded_sequence(char_embeddings, lengths, batch_first=True, enforce_sorted=False)
        
        # Run through LSTM
        lstm_outputs, _ = self.lstm(packed_embeddings)
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        
        # Split forward and backward outputs
        forward_outputs, backward_outputs = lstm_outputs.chunk(2, dim=-1)
        
        # Get predictions for forward and backward language models
        forward_logits = self.forward_proj(forward_outputs)
        backward_logits = self.backward_proj(backward_outputs)
        
        return {
            'forward_logits': forward_logits,
            'backward_logits': backward_logits,
            'representations': lstm_outputs
        }

    def get_elmo_representations(self, chars, lengths):
        # Get all layer representations
        outputs = self.forward(chars, lengths)
        representations = outputs['representations']
        
        # Compute normalized weights
        normed_weights = torch.softmax(self.scalar_parameters, dim=0)
        
        # Compute weighted sum
        weighted_sum = torch.zeros_like(representations)
        for i in range(len(normed_weights)):
            weighted_sum += normed_weights[i] * representations
            
        # Scale by gamma
        return self.gamma * weighted_sum

