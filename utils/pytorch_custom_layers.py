import torch.nn as nn
import torch

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


class STP(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_hidden_dim, num_layers, dropout, max_seq_length):
        super(STP, self).__init__()
        
        # Input Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Regression Head
        self.norm = nn.LayerNorm(embed_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 1)  # Output is a single regression value
        )
    
    def forward(self, x):
        # Input embedding with positional encoding
        batch_size, seq_length = x.size()
        embed = self.embedding(x) + self.positional_encoding[:, :seq_length, :]

        # Transformer blocks
        out = embed.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        for layer in self.layers:
            out = layer(out)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)

        # Apply LayerNorm over the embed_size dimension
        out = self.norm(out)  # Now out has shape (batch_size, seq_len, embed_size)

        # Max pooling over the sequence length
        out = out.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        out = self.max_pool(out).squeeze(-1)  # Now out has shape (batch_size, embed_size)

        # Regression Head
        out = self.mlp_head(out)
        return out
    

class STP_Tahn(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_hidden_dim, num_layers, dropout, max_seq_length):
        super(STP_Tahn, self).__init__()
        
        # Input Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Regression Head
        self.norm = nn.LayerNorm(embed_size)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(ff_hidden_dim, ff_hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(ff_hidden_dim // 2, 1)
        )

    def forward(self, x):
        # Input embedding with positional encoding
        batch_size, seq_length = x.size()
        embed = self.embedding(x) + self.positional_encoding[:, :seq_length, :]

        # Transformer blocks
        out = embed.permute(1, 0, 2)  # (seq_len, batch_size, embed_size)
        for layer in self.layers:
            out = layer(out)
        out = out.permute(1, 0, 2)  # (batch_size, seq_len, embed_size)

        # Apply LayerNorm over the embed_size dimension
        out = self.norm(out)  # Now out has shape (batch_size, seq_len, embed_size)

        # Max pooling over the sequence length
        out = out.permute(0, 2, 1)  # (batch_size, embed_size, seq_len)
        out = self.max_pool(out).squeeze(-1)  # Now out has shape (batch_size, embed_size)

        # Regression Head
        out = self.mlp_head(out)
        return out