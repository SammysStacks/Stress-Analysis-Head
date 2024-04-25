# General
import gc
import torch


def freeMemory():
    gc.collect()  # This line triggers Python's garbage collector. It will attempt to free up memory by collecting and disposing of objects that are no longer in use by the program.
    torch.cuda.empty_cache()  # This line clears the cached memory that PyTorch has allocated for the CUDA backend. It helps in freeing up unused GPU memory that PyTorch has reserved internally.

