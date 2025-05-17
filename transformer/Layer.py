import torch
import torch.nn as nn

from . import Module as M


class EncodingLayer(nn.Module):
    def __init__(
            self,
            d_model       : int,
            n_head        : int,
            layerNorm_eps : float = 1e-5,
            drop_rate     : float = 0.1
            ):
        super().__init__()

        self.SelfAttention = M.MultiHeadAttention(d_model = d_model, n_head = n_head)
        self.Linear        = nn.Linear(in_features = d_model, out_features = d_model)
        self.LayerNorm     = nn.LayerNorm(normalized_shape = d_model, eps = layerNorm_eps)
        self.Dropout       = nn.Dropout(drop_rate)

    def forward(
            self,
            x            : torch.Tensor,
            padding_mask : torch.Tensor = None
            ):
        """
        Parameters
        ----------
        x : shape = (batch_sze, seq_length, d_model)
        padding_mask : Padding mask, elements are ones or zeros, shape = (batch_size, seq_length)
        
        Return
        ------
        x_encoded : shape = (batch_sze, seq_length, d_model)
        """

        x_pre = x
        x = self.SelfAttention(x, x, x, padding_mask)
        x = self.LayerNorm(x + x_pre)

        x_pre = x
        x = self.Linear(x)
        x = self.LayerNorm(x + x_pre)

        x = self.Dropout(x)

        return x
    

class DecodingLayer(nn.Module):
    def __init__(
            self, 
            d_model       : int,
            n_head        : int,
            layerNorm_eps : float = 1e-5,
            drop_rate     : float = 0.1
            ):
        super().__init__()

        self.SelfAttention         = M.MultiHeadAttention(d_model = d_model, n_head = n_head)
        self.SourceTargetAttention = M.MultiHeadAttention(d_model = d_model, n_head = n_head)
        self.Linear                = nn.Linear(in_features = d_model, out_features = d_model)
        self.LayerNorm1            = nn.LayerNorm(normalized_shape = d_model, eps = layerNorm_eps)
        self.LayerNorm2            = nn.LayerNorm(normalized_shape = d_model, eps = layerNorm_eps)
        self.LayerNorm3            = nn.LayerNorm(normalized_shape = d_model, eps = layerNorm_eps)
        self.Dropout               = nn.Dropout(drop_rate)

    def forward(
            self,
            x            : torch.Tensor,
            x_encoded    : torch.Tensor,
            padding_mask : torch.Tensor = None,
            casual_mask  : torch.Tensor = None
            ):
        
        """
        Parameters
        ----------
        x : x from target, shape = (batch_sze, seq_length, d_model)
        x_encoded : Encoded x from encoder, shape = (batch_sze, seq_length, d_model)
        padding_mask : Padding mask, elements are ones or zeros, shape = (batch_size, seq_length)
        casual_mask  : Casual mask, elements are ones or zeros, shape = (batch_size, seq_length)

        Return
        ------
        x_decoded : shape = (batch_sze, seq_length, d_model)
        """

        x_pre = x
        x = self.SelfAttention(x, x, x, padding_mask, casual_mask)
        x = self.LayerNorm1(x + x_pre)

        x_pre = x
        x = self.SourceTargetAttention(x, x_encoded, x_encoded)
        x = self.LayerNorm2(x + x_pre)

        x_pre = x
        x = self.Linear(x)
        x = self.LayerNorm3(x + x_pre)

        x = self.Dropout(x)

        return x
        
