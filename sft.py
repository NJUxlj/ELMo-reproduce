import torch  
import torch.nn as nn  
from torch.utils.data import Dataset, DataLoader  
from config import Config  
from models.elmo import ELMo  
import logging  
from tqdm import tqdm  
import json  
import numpy as np  

from utils import load_vocab

class TaskSpecificModel(nn.Module):  
    '''
    任务特定的模型类，包含冻结的ELMo层和可训练的分类器层。
    '''
    def __init__(self, elmo_model, num_classes):  
        super().__init__()  
        self.elmo = elmo_model  
        # Freeze ELMo parameters  
        for param in self.elmo.parameters():  
            param.requires_grad = False  
            
        self.classifier = nn.Sequential(  
            nn.Linear(elmo_model.lstm.hidden_size * 2, 256),  
            nn.ReLU(),  
            nn.Dropout(0.1),  
            nn.Linear(256, num_classes)  
        )  
    
    def forward(self, chars, lengths):  
        # Get ELMo representations  
        elmo_output = self.elmo.get_elmo_representations(chars, lengths)  
        # Use mean pooling over sequence length  
        pooled = torch.mean(elmo_output, dim=1)  
        # Classify  
        return self.classifier(pooled)  

class TaskDataset(Dataset):  
    def __init__(self, data_path, char_vocab):  
        self.data = []  
        self.char_vocab = char_vocab  
        
        with open(data_path, 'r', encoding='utf-8') as f:  
            for line in f:  
                item = json.loads(line.strip())  
                self.data.append(item)  
    
    def __len__(self):  
        return len(self.data)  
    
    def __getitem__(self, idx):  
        item = self.data[idx]  
        chars = [[self.char_vocab.get(c, self.char_vocab['<UNK>']) for c in word]   
                for word in item['text']]  
        return {  
            'chars': chars,  
            'label': item['label']  
        }  

def collate_fn(batch):  
    max_word_len = max(max(len(word) for word in item['chars']) for item in batch)  
    max_seq_len = max(len(item['chars']) for item in batch)  
    
    batch_chars = torch.zeros(len(batch), max_seq_len, max_word_len).long()  
    lengths = []  
    labels = []  
    
    for i, item in enumerate(batch):  
        chars = item['chars']  
        lengths.append(len(chars))  
        labels.append(item['label'])  
        
        for j, word in enumerate(chars):  
            batch_chars[i, j, :len(word)] = torch.tensor(word)  
    
    return {  
        'chars': batch_chars,  
        'lengths': torch.tensor(lengths),  
        'labels': torch.tensor(labels)  
    }  

def finetune(num_classes, task_data_path):  
    config = Config()  
    
    # Setup logging  
    logging.basicConfig(  
        format='%(asctime)s - %(levelname)s - %(message)s',  
        level=logging.INFO,  
        handlers=[  
            logging.FileHandler(config.log_dir + 'finetuning.log'),  
            logging.StreamHandler()  
        ]  
    )  
    
    # Load pretrained ELMo model  
    checkpoint = torch.load(config.model_save_path)  
    
    char_vocab = load_vocab(config.char_vocab_path)    
    
    elmo_model = ELMo(  
        char_vocab_size=len(char_vocab), 
        hidden_size=config.hidden_size,  
        num_layers=config.num_layers,  
        dropout=config.dropout  
    ).to(config.device)  
    elmo_model.load_state_dict(checkpoint['model'])  
    
    # Create task-specific model  
    model = TaskSpecificModel(elmo_model, num_classes).to(config.device)  
    
    # Create datasets and dataloaders  
    train_dataset = TaskDataset(task_data_path + 'train.json', char_vocab)  
    val_dataset = TaskDataset(task_data_path + 'val.json', char_vocab)  
    
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=config.batch_size,  
        shuffle=True,  
        collate_fn=collate_fn  
    )  
    
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=config.batch_size,  
        shuffle=False,  
        collate_fn=collate_fn  
    )  
    
    # Setup optimizer and loss function  
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.finetune_lr)  
    criterion = nn.CrossEntropyLoss()  
    
    best_val_acc = 0  
    
    # Training loop  
    for epoch in range(config.finetune_epochs):  
        model.train()  
        total_loss = 0  
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:  
            for batch in pbar:  
                chars = batch['chars'].to(config.device)  
                lengths = batch['lengths']  
                labels = batch['labels'].to(config.device)  
                
                optimizer.zero_grad()  
                outputs = model(chars, lengths)  
                loss = criterion(outputs, labels)  
                
                loss.backward()  
                optimizer.step()  
                
                total_loss += loss.item()  
                pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})  
        
        # Validation  
        model.eval()  
        correct = 0  
        total = 0  
        
        with torch.no_grad():  
            for batch in val_loader:  
                chars = batch['chars'].to(config.device)  
                lengths = batch['lengths']  
                labels = batch['labels'].to(config.device)  
                
                outputs = model(chars, lengths)  
                _, predicted = torch.max(outputs.data, 1)  
                
                total += labels.size(0)  
                correct += (predicted == labels).sum().item()  
        
        val_acc = correct / total  
        logging.info(f'Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}')  
        
        # Save best model  
        if val_acc > best_val_acc:  
            best_val_acc = val_acc  
            torch.save({  
                'epoch': epoch,  
                'model_state_dict': model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),  
                'val_acc': val_acc  
            }, config.finetune_save_path)  
    
    logging.info(f'Best validation accuracy: {best_val_acc:.4f}')  
    return model  

if __name__ == '__main__':  
    # Example usage  
    num_classes = 2  # Binary classification  
    task_data_path = 'data/task/'  
    model = finetune(num_classes, task_data_path)