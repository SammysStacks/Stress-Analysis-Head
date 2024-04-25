
# -------------------------------------------------------------------------- #
# ---------------------------- Imported Modules ---------------------------- #

# General
import os
import sys
import tempfile
from datetime import timedelta

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


# -------------------------------------------------------------------------- #
# ---------------------- PyTorch DataLoader Interface ---------------------- #

class distributedGPUWrapper():
    
    def __init__(self, world_size, backends = "nccl", timeout = 600):        
        # General parameters
        self.timeout = timedelta(seconds=timeout)  # The total number of seconds to wait for GPU synchronization.
        self.world_size = world_size    # The total number of GPUs we are using.
        self.backends = backends        # The backend to run each GPU: 'nccl' (For GPUs), 'gloo' (For CPUs), 'mpi' (Experimental)
        
        # Broadcast the backend over each GPU if a single string is provided.
        if isinstance(self.backends, str): self.backends = [self.backends]*self.world_size
        
        # Assert the integrity of the input parameters.
        assert len(self.backends) == self.world_size, "You must specify a backend for each GPU"
        
    # ------------------------- Setup Distributions ------------------------ #
                
    def initializeProcess(self, rank):
        """ rank: The 0-indexed GPU we are using in the world_size """
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
    
        # initialize the process group
        dist.init_process_group(self.backends[rank], rank=rank, world_size=self.world_size, timeout = self.timeout)
    
    def initializeDistributions(self):
        # For each GPU we are considering.
        for rank in range(self.world_size):
            self.initializeProcess(rank)
            
    # ------------------------- Model Configuration ------------------------ #
    
    def addModelDistributions(self, model):
        # For each GPU we are considering.
        for rank in range(self.world_size):
            # create model and move it to GPU with id rank
            model = model.to(rank)
            ddp_model = DDP(model, device_ids=[rank])

    
    # ---------------------- Synchronize Distributions --------------------- #

    def cleanup(self):
        dist.destroy_process_group()
        
    # ---------------------------------------------------------------------- #
                            
# -------------------------------------------------------------------------- #
# --------------------------- Testing Information -------------------------- #

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


# -------------------------------------------------------------------------- #
# ------------------------------ User Testing ------------------------------ #

if __name__ == "__main__":    
    # Initialize your toy model
    model = ToyModel()
    

    
