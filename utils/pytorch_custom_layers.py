"""
This module defines custom layers and models for PyTorch.

Classes:
--------
- TransformerBlock: Implements a single transformer block with multi-head attention and feed-forward layers.
- LogPPredictionModel: Combines a base model with transformer blocks and an MLP head for regression tasks.

Authors:
--------
Timo Ehrle & Levin Willi

Last Modified:
--------------
10.12.2024
"""

# import necessary packages
import torch.nn as nn
#=======================================================================================================================

class TransformerBlock(nn.Module):
    """
    Implements a single transformer block with multi-head attention, feed-forward layers,
    LayerNorm and dropout.

    Attributes:
    -----------
    attention : torch.nn.MultiheadAttention
        Multi-head self-attention mechanism.
    norm1 : torch.nn.LayerNorm
        Layer normalization applied after the attention mechanism.
    norm2 : torch.nn.LayerNorm
        Layer normalization applied after the feed-forward network.
    ffn : torch.nn.Sequential
        Feed-forward network consisting of two linear layers with a ReLU activation in between.
    dropout : torch.nn.Dropout
        Dropout layer applied to prevent overfitting.

    Methods:
    --------
    forward(x):
        Performs a forward pass through the transformer block.
    """
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout):
        """
        Initializes the transformer block.

        Parameters:
        -----------
        embed_size : int
            Dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads in the multi-head attention mechanism.
        ff_hidden_dim : int
            Dimensionality of the hidden layer in the feed-forward network.
        dropout : float
            Dropout rate to be applied in attention and feed-forward layers.
        """
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
        """
        Performs a forward pass through the transformer block.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (seq_len, batch_size, embed_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (seq_len, batch_size, embed_size) after processing
            through the transformer block.
        """
        # Attention layer with residual connection and LayerNorm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward layer with residual connection and LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
    
class LogPPredictionModel(nn.Module):
    """
    Combines a base model with transformer blocks and an MLP head for regression tasks.

    Attributes:
    -----------
    base_model : torch.nn.Module
        Pretrained model providing contextual embeddings (e.g., a transformer encoder).
    transformer_layers : torch.nn.ModuleList
        A stack of custom transformer blocks for further sequence modeling.
    norm : torch.nn.LayerNorm
        Layer normalization applied to the transformer output.
    max_pool : torch.nn.AdaptiveMaxPool1d
        Adaptive max pooling layer applied over the sequence length dimension.
    mlp_head : torch.nn.Sequential
        A multi-layer perceptron head for regression.

    Methods:
    --------
    forward(input_ids, attention_mask):
        Performs a forward pass through the combined model.
    """
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
        """
        Initializes the LogPPredictionModel.

        Parameters:
        -----------
        base_model : torch.nn.Module
            Pretrained model providing contextual embeddings.
        embed_size : int
            Dimensionality of the input embeddings.
        num_heads : int
            Number of attention heads in the transformer blocks.
        ff_hidden_dim : int
            Dimensionality of the hidden layer in the feed-forward network.
        num_layers : int
            Number of transformer blocks to stack.
        dropout : float
            Dropout rate to be applied in attention and feed-forward layers.
        """
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