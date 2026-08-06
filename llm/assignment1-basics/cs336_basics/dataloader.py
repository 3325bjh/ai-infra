import torch
import numpy as np
import numpy.typing as npt
def get_batch(
        dataset:npt.NDArray,
        batch_size:int,
        max_seq_length:int,
        device:str
)->tuple[torch.Tensor,torch.Tensor]:
    n=len(dataset)
    max_idx=n-max_seq_length-1
    ix=torch.randint(0,max_idx+1,(batch_size,))
    x=torch.stack([torch.from_numpy(dataset[i:i+max_seq_length].astype(np.int64)) for i in ix])
    y=torch.stack([torch.from_numpy(dataset[i+1:i+max_seq_length+1].astype(np.int64)) for i in ix])
    return x.to(device),y.to(device)
