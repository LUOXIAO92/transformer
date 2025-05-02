import torch



def __getMaskInfo(
        mask : torch.Tensor
        ):
    n_batch, seq_length = mask.shape

    mask_starts = [len(mask[i,:][mask[i,:] == 1]) for i in range(n_batch)]
    
    mask_mat = [torch.zeros(size  = (seq_length, seq_length), 
                            dtype = torch.float) 
                for i in range(n_batch)]
    
    return n_batch, seq_length, mask_starts, mask_mat

def casual_mask(
        mask : torch.Tensor,
        ) -> torch.Tensor:
    """
    Parameters
    ----------
    mask : Mask vectors, shape = (n_batch, seq_length)

    Return
    ------
    mask_matrix : Mask matrices, shape = (n_batch, seq_length, seq_length)
    """
    
    n_batch, seq_length, mask_starts, mask_mat = __getMaskInfo(mask)

    mask_mat_part = [
        torch.triu(
            torch.full(size       = (seq_length - mask_starts[i] - 1, seq_length - mask_starts[i] - 1), 
                       fill_value = -torch.inf,
                       dtype      = float)
            ) for i in range(n_batch) ]

    for i in range(n_batch):
        mask_mat[:seq_length - mask_starts[i] - 1, mask_starts[i] + 1:] = mask_mat_part[i]

    return torch.as_tensor(mask_mat)

def padding_mask(
        mask : torch.Tensor,
        ):
    """
    Parameters
    ----------
    mask : Mask vectors, shape = (n_batch, seq_length)

    Return
    ------
    mask_matrix : Mask matrices, shape = (n_batch, seq_length, seq_length)
    """

    n_batch, seq_length, mask_starts, mask_mat = __getMaskInfo(mask)

    src_mask = torch.einsum('bl,bL->blL', mask, mask)
    src_mask = (src_mask == 0)
    mask_mat[src_mask] = -torch.inf

    return torch.as_tensor(mask_mat)