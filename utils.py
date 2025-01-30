import torch  
import numpy as np  
import os  
import logging  
import json  
from typing import List, Dict, Tuple, Optional  
from collections import Counter  

def setup_logging(log_dir: str, log_filename: str) -> None:  
    """  
    设置日志配置  
    
    Args:  
        log_dir: 日志目录  
        log_filename: 日志文件名  
    """  
    if not os.path.exists(log_dir):  
        os.makedirs(log_dir)  
        
    logging.basicConfig(  
        format='%(asctime)s - %(levelname)s - %(message)s',  
        level=logging.INFO,  
        handlers=[  
            logging.FileHandler(os.path.join(log_dir, log_filename)),  
            logging.StreamHandler()  
        ]  
    )  

def build_char_vocab(data_files: List[str], min_freq: int = 2) -> Dict[str, int]:  
    """  
    从数据文件构建字符级词表  
    
    Args:  
        data_files: 数据文件路径列表  
        min_freq: 最小频率阈值  
        
    Returns:  
        字符到索引的映射字典  
    """  
    char_counter = Counter()  
    
    # 统计字符频率  
    for file_path in data_files:  
        with open(file_path, 'r', encoding='utf-8') as f:  
            for line in f:  
                text = json.loads(line.strip())['text']  
                for word in text:  
                    for char in word:  
                        char_counter[char] += 1  
    
    # 构建词表  
    char_vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}  
    for char, freq in char_counter.items():  
        if freq >= min_freq:  
            char_vocab[char] = len(char_vocab)  
    
    return char_vocab  

def save_vocab(vocab: Dict[str, int], vocab_path: str) -> None:  
    """  
    保存词表到文件  
    
    Args:  
        vocab: 词表字典  
        vocab_path: 保存路径  
    """  
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)  
    with open(vocab_path, 'w', encoding='utf-8') as f:  
        json.dump(vocab, f, ensure_ascii=False, indent=2)  

def load_vocab(vocab_path: str) -> Dict[str, int]:  
    """  
    从文件加载词表  
    
    Args:  
        vocab_path: 词表文件路径  
        
    Returns:  
        词表字典  
    """  
    with open(vocab_path, 'r', encoding='utf-8') as f:  
        return json.load(f)  

def pad_sequences(sequences: List[List[int]],   
                 max_len: Optional[int] = None,   
                 padding_value: int = 0) -> torch.Tensor:  
    """  
    对序列进行填充  
    
    Args:  
        sequences: 序列列表  
        max_len: 最大长度，如果为None则使用最长序列的长度  
        padding_value: 填充值  
        
    Returns:  
        填充后的张量  
    """  
    if max_len is None:  
        max_len = max(len(seq) for seq in sequences)  
    
    padded_seqs = []  
    for seq in sequences:  
        padded_seq = seq[:max_len] + [padding_value] * max(0, max_len - len(seq))  
        padded_seqs.append(padded_seq)  
    
    return torch.tensor(padded_seqs)  

def create_char_sequences(texts: List[List[str]],   
                        char_vocab: Dict[str, int],  
                        max_word_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:  
    """  
    将文本转换为字符级序列  
    
    Args:  
        texts: 单词列表的列表  
        char_vocab: 字符词表  
        max_word_len: 最大单词长度  
        
    Returns:  
        (char_sequences, word_lengths)  
    """  
    char_sequences = []  
    word_lengths = []  
    
    for text in texts:  
        chars = []  
        lengths = []  
        for word in text:  
            word_chars = [char_vocab.get(c, char_vocab['<UNK>']) for c in word]  
            chars.append(word_chars)  
            lengths.append(len(word_chars))  
        char_sequences.append(chars)  
        word_lengths.append(lengths)  
    
    # Pad character sequences  
    max_seq_len = max(len(seq) for seq in char_sequences)  
    if max_word_len is None:  
        max_word_len = max(max(len(word) for word in seq) for seq in char_sequences)  
    
    batch_size = len(char_sequences)  
    padded_chars = torch.zeros(batch_size, max_seq_len, max_word_len).long()  
    
    for i, seq in enumerate(char_sequences):  
        for j, word in enumerate(seq):  
            padded_chars[i, j, :len(word)] = torch.tensor(word[:max_word_len])  
    
    return padded_chars, torch.tensor(word_lengths)  

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:  
    """  
    计算准确率  
    
    Args:  
        predictions: 模型预测结果  
        labels: 真实标签  
        
    Returns:  
        准确率  
    """  
    _, predicted = torch.max(predictions, 1)  
    correct = (predicted == labels).sum().item()  
    total = labels.size(0)  
    return correct / total  

def save_metrics(metrics: Dict, save_path: str) -> None:  
    """  
    保存评估指标  
    
    Args:  
        metrics: 评估指标字典  
        save_path: 保存路径  
    """  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    with open(save_path, 'w', encoding='utf-8') as f:  
        json.dump(metrics, f, indent=2)  

def load_metrics(metrics_path: str) -> Dict:  
    """  
    加载评估指标  
    
    Args:  
        metrics_path: 指标文件路径  
        
    Returns:  
        评估指标字典  
    """  
    with open(metrics_path, 'r', encoding='utf-8') as f:  
        return json.load(f)  

def set_seed(seed: int) -> None:  
    """  
    设置随机种子以确保可重复性  
    
    Args:  
        seed: 随机种子  
    """  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  

def count_parameters(model: torch.nn.Module) -> int:  
    """  
    统计模型参数量  
    
    Args:  
        model: PyTorch模型  
        
    Returns:  
        可训练参数数量  
    """  
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:  
    """  
    获取当前学习率  
    
    Args:  
        optimizer: PyTorch优化器  
        
    Returns:  
        当前学习率  
    """  
    for param_group in optimizer.param_groups:  
        return param_group['lr']