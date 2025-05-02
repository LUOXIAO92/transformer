import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe

import Function

class PositionEncoding(nn.Module):
    def __init__(
            self
            ):
        super().__init__()


    def forward(
            self,
            x : torch.Tensor
            ):
        
        """
        Parameters
        ----------
        x : Shape = (batch_size, seq_length, d_model)

        Return
        ------
        POS : Position encoding vector for each corresponding batches and tokens. Shape = (batch_size, seq_length, d_model)
        """

        batch_size, seq_length, d_model = x.shape
        
        wave = [torch.tensor([10000 ** (i / d_model), 10000**((i + 1) / d_model)], dtype = x.dtype) for i in range(0, d_model, 2)]
        wave = torch.concat(wave, dim = 0)
        Iwave = torch.ones_like(wave)

        pos = torch.arange(0, seq_length, 1, dtype = x.dtype)
        Ipos = torch.ones_like(pos)

        onePE = oe.contract('i,j->ij', pos, Iwave) / oe.contract('i,j->ij', Ipos, wave)
        onePE[:,0::2] = torch.sin(onePE[:,0::2])
        onePE[:,1::2] = torch.cos(onePE[:,1::2])

        PE = torch.zeros_like(x)
        for i in range(batch_size):
            PE[i,:,:] = onePE

        x += PE

        return x

class ScaleProductAttention(nn.Module):
    def __init__(
            self, 
            dk : int
            ):
        super().__init__(nn.Module)

        self.dk = dk

        self.cal_Q = nn.Linear(in_features = dk, out_features = dk)
        self.cal_K = nn.Linear(in_features = dk, out_features = dk)
        self.cal_V = nn.Linear(in_features = dk, out_features = dk)

    def forward(
            self, 
            xQ           : torch.Tensor, 
            xK           : torch.Tensor, 
            xV           : torch.Tensor, 
            padding_mask : torch.Tensor = None,
            casual_mask  : torch.Tensor = None
            ) -> torch.Tensor:
        
        """
        Parameters
        ----------
        xQ : x for Query, shape = (batch_size, query_seq_len, d_k)
        xK : x for Key  , shape = (batch_size, key_seq_len  , d_k)
        xV : x for Value, shape = (batch_size, value_seq_len, d_k)
        padding_mask : Mask for padding        , shape = (batch_size, seq_len)
        casual_mask  : Mask for target sequence, shape = (batch_size, seq_len)

        Returns
        -------
        x : Attention(Q,K,V) with mask with the shape of (batch_size, query_seq_len, d_k).
        """
        Q = self.cal_Q(xQ)
        K = self.cal_K(xK)
        V = self.cal_V(xV)

        QKT = oe.contract('bQd,bKd->bQK', Q, K)

        mask_matrix = None
        if padding_mask is not None:
            mask_matrix = Function.padding_mask(padding_mask)

        if casual_mask is not None:
            if mask_matrix is not None:
                mask_matrix += Function.casual_mask(casual_mask)
            else:
                mask_matrix = Function.casual_mask(casual_mask)

        if mask_matrix is not None:
            QKT += mask_matrix

        S = F.softmax(QKT, dim = -1)
        del QKT
        Attention = oe.contract('bQK,Kd->bQd', S, V)

        return Attention


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            d_model        : int,
            n_head         : int,
            ):
        super().__init__()

        self.d_model = d_model
        self.n_head  = n_head
        self.d_k = d_model // n_head

        self.SDPAs = nn.ModuleList(
            [ScaleProductAttention(dk = self.dk) 
             for i in range(n_head)]
            )

        self.WOs = nn.ModuleList(
            [nn.Linear(in_features = self.d_k, out_features = self.d_model) 
             for i in range(n_head)]
                                 )
        #self.WO = nn.Parameter(torch.rand(size = (n_head, self.d_k, d_model)))
        
    def forward(
            self,
            xQ           : torch.Tensor,
            xK           : torch.Tensor,
            xV           : torch.Tensor,
            padding_mask : torch.Tensor = None,
            casual_mask  : torch.Tensor = None
            ):
        
        """
        Parameters
        ----------
        xQ : x for query, shape = (batch_size, query_seq_len, d_model)
        xK : x for key  , shape = (batch_size, key_seq_len  , d_model)
        xV : x for value, shape = (batch_size, value_seq_len, d_model)
        padding_mask : Mask for padding        , shape = (batch_size, seq_len)
        casual_mask  : Mask for target sequence, shape = (batch_size, sqc_len)

        Returns
        -------
        x : Attention(Q,K,V) with mask.
        """
        
        batch_size = xQ.shape[0]
        q_seq_len  = xQ.shape[1]
        k_seq_len  = xK.shape[1]
        v_seq_len  = xV.shape[1]
        
        As = []
        for SDPA in self.SDPAs:
            As.append(
                SDPA.forward(xQ, xK, xV, padding_mask, casual_mask)
                )
            
        Attention = torch.zeros(size = (q_seq_len, self.d_model))
        for A, WO in As, self.WOs:
            Attention += WO(A)

        del As

        return Attention
    

