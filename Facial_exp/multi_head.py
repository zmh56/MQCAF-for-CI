# ref in https://github.com/d2l-ai/d2l-zh
# @Time    : 27/4/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university


import math
import torch
from torch import nn

# Function to reshape query, key, and value tensors for multi-head attention
def transpose_qkv(X, num_heads):
    """Transform the shape of X for parallel multi-head attention."""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)  # Reshaping
    X = X.permute(0, 2, 1, 3)  # Permuting to correct shape for attention
    return X.reshape(-1, X.shape[2], X.shape[3])  # Flattening the batch and num_heads

# Reverse operation to undo the transformation from transpose_qkv
def transpose_output(X, num_heads):
    """Revert the operation of transpose_qkv."""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# Mask irrelevant entries in sequences based on valid lengths
def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# Perform masked softmax to ignore padded positions
def masked_softmax(X, valid_lens):
    """Softmax operation with masking."""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)  # Masking invalid entries
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# Scaled Dot-Product Attention mechanism
class DotProductAttention(nn.Module):
    """Scaled dot-product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """Calculate attention scores and apply to values."""
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # Calculate raw attention scores
        self.attention_weights = masked_softmax(scores, valid_lens)  # Apply softmax and masking
        return torch.bmm(self.dropout(self.attention_weights), values)  # Output the weighted values

# Multi-Head Attention mechanism that applies multiple attention heads in parallel
class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """Compute multi-head attention."""
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# Main script to initialize and test the multi-head attention mechanism
if __name__ == "__main__":
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    attention.eval()  # Set the model to evaluation mode

    # Test data with random valid lengths
    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))  # Sample input
    output = attention(X, X, X, valid_lens)  # Compute attention
    print(output.shape)  # Print output shape to verify correct operation
