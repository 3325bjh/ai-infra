import os
import typing

import torch
def save_checkpoint(
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        iteration:int,
        out:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]]
):
    checkpoint={
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(checkpoint,out)

def load_checkpoint(
        src:typing.Union[str,os.PathLike,typing.BinaryIO,typing.IO[bytes]],
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer
):
    checkpoint=torch.load(src,map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]