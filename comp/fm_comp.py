import torch
import torch.nn as nn


class Compressor_Decompressor(nn.Module):
    def __init__(self, dim, compressed_dim=64, drop=0.):
        super(CompressDecompress, self).__init__()
        
        self.compressor = nn.Linear(dim, compressed_dim)
        self.decompressor = nn.Linear(compressed_dim, dim)
        self.compressor_drop = nn.Dropout(drop)
        self.decompressor_drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.compressor_drop(self.compressor(x))
        x = self.decompressor_drop(self.decompressor(x))
        return x

