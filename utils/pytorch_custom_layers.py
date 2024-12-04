import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention layer with residual connection and LayerNorm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward layer with residual connection and LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
    
class LogPPredictionModel(nn.Module):
    def __init__(self, base_model, embed_size, num_heads, ff_hidden_dim, num_layers, dropout):
        super(LogPPredictionModel, self).__init__()
        self.base_model = base_model
        
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Normalization and pooling
        self.norm = nn.LayerNorm(embed_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 1)  # Regression output
        )
    
    def forward(self, input_ids, attention_mask):
        # Forward pass through the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Permute for Transformer processing
        x = hidden_states.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        
        # Forward pass through Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Permute back to (batch_size, seq_len, embed_size)
        x = x.permute(1, 0, 2)
        
        # Apply LayerNorm
        x = self.norm(x)
        
        # Max pooling over the sequence length
        x = x.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        x = self.max_pool(x).squeeze(-1)  # (batch_size, embed_size)
        
        # MLP Head for regression
        output = self.mlp_head(x)
        return output