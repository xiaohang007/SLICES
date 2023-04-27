import torch
import numpy as np

def Variable(tensor):
    """
    Wrapper function to generate torch tensor and assign to GPU if available
    Args:
        tensor: input tensor

    Returns: torch tensor

    """
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
        tensor.requires_grad = True
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor 


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """

    Args:
        optimizer: optimizer for the DL model
        decrease_by: multiplies the learning rate by (1-decrease_by)

    Returns: None

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1 - decrease_by)


def seq_to_smiles(seqs, voc):
    """
    Takes an output sequence from RNN and returns the smiles
    Args:
        seqs:
        voc:

    Returns: SMILE string

    """
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles




def unique(arr):
    """
    Find unique rows in arr and return their indices
    Args:
        arr:

    Returns: Indices of unique rows

    """
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize*arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))
