import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention with residual connection
        attention_output, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + self.dropout(attention_output))
        
        # Feed-forward network with residual connection
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward_output))
        
        return x