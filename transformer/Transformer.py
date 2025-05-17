import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Module
from . import Layer

class Encoder(nn.Module):
    def __init__(
            self,
            N_encodinglayer : int,
            d_model         : int,
            n_head          : int,
            layerNorm_eps   : float = 1e-5,
            drop_rate       : float = 0.1
            ):
        super().__init__()

        self.EncodingLayers = nn.ModuleList(
            [Layer.EncodingLayer(
                d_model       = d_model      , 
                n_head        = n_head       , 
                layerNorm_eps = layerNorm_eps, 
                drop_rate     = drop_rate    
                ) 
             for _ in range(N_encodinglayer)]
        )

    def forward(
            self,
            x            : torch.Tensor,
            padding_mask : torch.Tensor
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

        for layer in self.EncodingLayers:
            x = layer(x, padding_mask)

        return x
    
class Decoder(nn.Module):
    def __init__(
            self,
            N_decodinglayer : int,
            d_model         : int,
            n_head          : int,
            layerNorm_eps   : float = 1e-5,
            drop_rate       : float = 0.1
            ):
        super().__init__()

        self.DecodingLayers = nn.ModuleList(
            [Layer.DecodingLayer(
                d_model       = d_model      , 
                n_head        = n_head       , 
                layerNorm_eps = layerNorm_eps, 
                drop_rate     = drop_rate    
                ) 
             for _ in range(N_decodinglayer)]
        )

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

        for layer in self.DecodingLayers:
            x = layer(x, x_encoded, padding_mask, casual_mask) 

        return x
    

class Transformer(nn.Module):
    def __init__(
            self,
            vocab_size      : int,
            padding_idx     : int,
            N_encodinglayer : int,
            N_decodinglayer : int,
            d_model         : int,
            n_head          : int,
            layerNorm_eps   : float = 1e-5,
            drop_rate       : float = 0.1
            ):
        super().__init__()

        self.PositionalEmbedding = Module.PositionEncoding()

        self.Embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = d_model,
            padding_idx    = padding_idx
            )
        
        self.Linear = nn.Linear(
            in_features  = d_model,
            out_features = vocab_size
            )

        self.Encoder = Encoder(
            N_encodinglayer = N_encodinglayer,
            d_model         = d_model,
            n_head          = n_head,
            layerNorm_eps   = layerNorm_eps,
            drop_rate       = drop_rate
            )

        self.Decoder = Decoder(
            N_decodinglayer = N_decodinglayer,
            d_model         = d_model,
            n_head          = n_head,
            layerNorm_eps   = layerNorm_eps,
            drop_rate       = drop_rate
            )
        
    def forward(
            self,
            src : torch.Tensor,
            tgt : torch.Tensor,
            src_padding_mask : torch.Tensor = None,
            tgt_padding_mask : torch.Tensor = None,
            tgt_casual_mask  : torch.Tensor = None
            ):
        """
        Parameters
        ----------
        src : Source sequences, shape = (batch_size, src_seq_length)
        tgt : Target sequences, shape = (batch_size, tgt_seq_length)
        src_padding_mask : Padding mask for source sequences, shape = (batch_size, src_seq_length)
        tgt_padding_mask : Padding mask for target sequences, shape = (batch_size, tgt_seq_length)
        tgt_casual_mask  : Casual mask for target sequences, shape = (batch_size, tgt_seq_length)
         
        Return
        ------
        output : Output of the `Logit` value, shape = (batch_size, out_seq_length, d_model)
        """
        
        source = self.Embedding(src)
        source = self.PositionalEmbedding(source)
        source = self.Encoder(source, src_padding_mask)

        target = self.Embedding(tgt)
        target = self.PositionalEmbedding(target)
        target = self.Decoder(source, target, tgt_padding_mask, tgt_casual_mask)

        output = self.Linear(target)

        return output