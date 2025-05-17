import torch



def __getMaskInfo(
        mask : torch.Tensor
        ):
    if mask.ndim < 2:
        raise RuntimeError("Dimension of mask must be (batch_size, seq_length)")
    batch_size, seq_length = mask.shape

    mask_starts = [len(mask[i,:][mask[i,:] == 1]) for i in range(batch_size)]
    
    mask_mat = torch.zeros(size = (batch_size, seq_length, seq_length))
    for b in range(batch_size):
        mask_mat[b] = torch.zeros(
            size  = (seq_length, seq_length), 
            dtype = torch.float
            ) 
    
    return batch_size, seq_length, mask_starts, mask_mat

def casual_mask(
        mask : torch.Tensor,
        ) -> torch.Tensor:
    """
    Parameters
    ----------
    mask : Mask vectors, shape = (batch_size, seq_length)

    Return
    ------
    mask_matrix : Mask matrices, shape = (batch_size, seq_length, seq_length)
    """
    
    batch_size, seq_length, mask_starts, mask_mat = __getMaskInfo(mask)
    
    mask_mat_part = [
        torch.triu(
            torch.full(
                size       = (seq_length - mask_starts[i], seq_length - mask_starts[i]), 
                fill_value = -torch.inf,
                dtype      = float
                )
            ) for i in range(batch_size) ]

    for i in range(batch_size):
        mask_mat[i][:seq_length - mask_starts[i], mask_starts[i]:] = mask_mat_part[i]

    return mask_mat

def padding_mask(
        mask : torch.Tensor,
        ):
    """
    Parameters
    ----------
    mask : Mask vectors, shape = (batch_size, seq_length)

    Return
    ------
    mask_matrix : Mask matrices, shape = (batch_size, seq_length, seq_length)
    """

    batch_size, seq_length, mask_starts, mask_mat = __getMaskInfo(mask)

    src_mask = torch.einsum('bl,bL->blL', mask, mask)
    mask_mat[src_mask == 0] = -torch.inf

    return torch.as_tensor(mask_mat)