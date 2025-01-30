
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class Config:
    # Data parameters
    train_data_path = "data/train.txt"
    valid_data_path = "data/valid.txt"
    char_vocab_path = "data/char_vocab.txt"
    
    # Model parameters
    char_embed_dim = 16
    hidden_size = 512
    num_layers = 2
    dropout = 0.1
    
    # Training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    max_grad_norm = 5.0
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Save and load
    model_save_path = "checkpoints/elmo_model.pt"
    log_dir = "logs/"

