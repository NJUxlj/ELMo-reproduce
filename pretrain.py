
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import Config
from models.elmo import ELMo
from utils import create_char_vocab, load_data
from tqdm import tqdm
import logging

class TextDataset(Dataset):
    def __init__(self, texts, char_vocab):
        self.texts = texts
        self.char_vocab = char_vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convert text to character indices
        char_indices = [[self.char_vocab.get(c, self.char_vocab['<unk>']) for c in word] 
                       for word in text.split()]
        return torch.tensor(char_indices)

def collate_fn(batch):
    # Pad sequences
    lengths = torch.tensor([len(x) for x in batch])
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths

def train():
    config = Config()
    
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(config.log_dir + 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create character vocabulary
    char_vocab = create_char_vocab(config.train_data_path)
    
    # Load data
    train_texts = load_data(config.train_data_path)
    valid_texts = load_data(config.valid_data_path)
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_texts, char_vocab)
    valid_dataset = TextDataset(valid_texts, char_vocab)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = ELMo(
        char_vocab_size=len(char_vocab),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(config.device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    best_valid_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch, lengths in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}'):
            batch = batch.to(config.device)
            lengths = lengths.to(config.device)
            
            optimizer.zero_grad()
            
            outputs = model(batch, lengths)
            
            # Calculate forward and backward language model losses
            forward_loss = criterion(
                outputs['forward_logits'][:, :-1].reshape(-1, len(char_vocab)),
                batch[:, 1:].reshape(-1)
            )
            
            backward_loss = criterion(
                outputs['backward_logits'][:, 1:].reshape(-1, len(char_vocab)),
                batch[:, :-1].reshape(-1)
            )
            
            loss = forward_loss + backward_loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        
        with torch.no_grad():
            for batch, lengths in valid_loader:
                batch = batch.to(config.device)
                lengths = lengths.to(config.device)
                
                outputs = model(batch, lengths)
                
                forward_loss = criterion(
                    outputs['forward_logits'][:, :-1].reshape(-1, len(char_vocab)),
                    batch[:, 1:].reshape(-1)
                )
                
                backward_loss = criterion(
                    outputs['backward_logits'][:, 1:].reshape(-1, len(char_vocab)),
                    batch[:, :-1].reshape(-1)
                )
                
                valid_loss += (forward_loss + backward_loss).item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_valid_loss = valid_loss / len(valid_loader)
        
        logging.info(f'Epoch {epoch+1}/{config.num_epochs}:')
        logging.info(f'Average training loss: {avg_train_loss:.4f}')
        logging.info(f'Average validation loss: {avg_valid_loss:.4f}')
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_loss': best_valid_loss,
            }, config.model_save_path)
            logging.info(f'Model saved to {config.model_save_path}')

if __name__ == "__main__":
    train()

