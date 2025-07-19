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
        #Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        #LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            batch_first=True, 
                            dropout=dropout if dropout > 0 else 0,
                            bidirectional=True)
        #Fully-Connected Layer                    
        self.fc = nn.Linear(hidden_dim * 2, num_labels)
        #CRF Layer
        self.crf = CRF(num_labels, batch_first=True)
        """
        #CRF层的作用是保证输出序列的合法性，即保证输出序列中相邻的标签是合法的
        #例如，如果标签是B-PER，那么下一个标签不能是I-PER，只能是非实体标签
        """
        #Padding Index
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
            # mask的作用是：在计算CRF损失时，忽略padding位置的预测
            mask = (input_ids != self.pad_idx)
        
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask.bool(), reduction='mean')
            return loss
        else:
            best_path = self.crf.decode(emissions, mask=mask.bool())
            # 严格类型保护，保证返回list of list
            if isinstance(best_path, list):
                if len(best_path) == 0:
                    return []
                elif isinstance(best_path[0], int):
                    return [best_path]
                else:
                    return best_path
            else:
                return [[best_path]]
