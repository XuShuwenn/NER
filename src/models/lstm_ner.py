import torch
import torch.nn as nn
from torchcrf import CRF

class LSTMNER(nn.Module):
    """LSTM NER模型类框架"""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 num_labels,
                 pad_idx=0, 
                 embedding_matrix=None,
                 dropout=0.1):
        """初始化LSTM NER模型框架"""
        super(LSTMNER, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            batch_first=True, 
                            dropout=dropout if dropout > 0 else 0,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)#Fully-Connected Layer
        self.crf = CRF(num_labels, batch_first=True)#Conditional Random Field Layer
        self.pad_idx = pad_idx #Padding Index
        
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = False
            
    def forward(self, input_ids, labels=None, mask=None):
        """前向传播框架"""
        embeds = self.embedding(input_ids)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
              
        if mask is None:
            # mask: (batch_size, seq_len), 1 for real tokens, 0 for padding
            mask = (input_ids != self.pad_idx)
        
        if labels is not None:
            # CRF expects mask to be bool
            loss = -self.crf(emissions, labels, mask=mask.bool(), reduction='mean')
            return loss
        else:
            # decode returns best path (batch_size, seq_len)
            best_path = self.crf.decode(emissions, mask=mask.bool())
            return best_path
