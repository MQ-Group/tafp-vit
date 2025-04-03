import torch
import torch.nn as nn

def inter_layer_token_pruning(attn, num_token_pruning):
    '''
    input:
        attn: (B, num_heads, N, N)
        num_token_pruning; int
    output:
        token_pruning_mask: (B, N)
    '''

    B, H, N, _ = attn.shape
    attn_column_sum = torch.sum(attn,dim=(-2,-3))
    token_rank = torch.sort(attn_column_sum, dim=-1, descending=True).indices
    token_pruning_mask = attn_rank >= num_token_pruning

    return token_pruning_mask

def reshape_attn(token_pruning_mask, attn):
    '''
    input:
        token_pruning_mask: (B, N)
        attn: (B, num_heads, N, N)
    output:
        attn: (B, num_heads, N-num_token_pruning, N-num_token_pruning)
    '''
    B, H, N, _ = attn.shape
    attn_pruning_mask = token_pruning_mask.view([B, 1, N, 1]).repeat([1, H, 1, N]) & token_pruning_mask.view([B, 1, 1, N]).repeat([1, H, N, 1])
    N -= num_token_pruning
    attn = attn[attn_pruning_mask].reshape(B, H, N, N)

    return attn

def reshape_v(token_pruning_mask, v):
    '''
    input:
        token_pruning_mask: (B, N)
        v: (B, num_heads, N, C // self.num_heads)
    output:
        v: (B, num_heads, N-num_token_pruning, C // self.num_heads)
    '''

    B, H, N, C = v.shape
    v = v[token_pruning_mask.view([B, 1, N, 1]).repeat([1, H, 1, C])].reshape(B, H, -1, C)

    return v

def reshape_x(token_pruning_mask, v):
    '''
    input:
        token_pruning_mask: (B, N)
        x: (B, N, C)
    output:
        x: (B, N-num_token_pruning, C)
    '''

    B, N, C = x.shape
    x = x[token_pruning_mask.view([B, N, 1]).repeat([1, 1, C])].reshape(B, -1, C)

    return x


def intra_block_row_pruning(attn, num_block_row_pruning):
    '''
    input:
        attn: (B, num_heads, N, N)
        num_block_row_pruning; (num_heads, 4, 4, 1)
    output:
        attn: (B, num_heads, N, N)
    '''
    
    num_block_row_pruning = num_block_row_pruning[:, :(N+63)//64, :, :][:, :, :(N+63)//64, :]

    B, H, N, _ = attn.shape

    if N % 64 != 0:
        attn = F.pad(attn, (0, 64 - N % 64, 0, 64 - N % 64), mode='constant', value=0)

    attn = attn.unfold(dimension=3, size=64, step=64)

    attn = attn.unfold(dimension=2, size=64, step=64)

    attn = attn.permute(0, 1, 2, 3, 5, 4)

    block_row_abs_sum = torch.sum(torch.abs(attn), dim=-1)
    block_row_rank = torch.sort(block_row_abs_sum, dim=-1, descending=True).indices
    block_row_pruning_mask = block_row_rank >= num_block_row_pruning

    block_row_pruning_mask = block_row_pruning_mask.unsqueeze(-1).repeat([1, 1, 1, 1, 1, 64])
    block_row_pruning_mask = block_row_pruning_mask.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, H, (N+63)//64*64, (N+63)//64*64)

    if N % 64 != 0:
        block_row_pruning_mask = block_row_pruning_mask[:, :, :N, :][:, :, :, :N]

    attn[block_row_pruning_mask] = 0
    return attn







        



