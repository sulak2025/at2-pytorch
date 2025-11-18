from model import Net
import torch
import os
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup(rank: int, world_size: int):
   """
   Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)  # sets the default GPU for each process
  #Using NCCL(NVIDIA Collective Communications Library (NCCL)) backend for distributed training with CUDA GPU
   init_process_group(backend="nccl", rank=rank, world_size=world_size) 

#model = net

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    # 1️⃣ Load CIFAR-10 training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 2️⃣ Initialize your model (replace Net() with your actual model class)
    net = Net()
    net.load_state_dict(torch.load('./cifar_net.pth'))  # load trained weights
    model = net

    # 3️⃣ Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, rank: int, world_size: int):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # shuffling done here per epoch
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=sampler
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size, rank, world_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    # Training settings (set manually for notebook)
    total_epochs = 10     # total number of epochs
    save_every = 2        # how often to save checkpoints
    batch_size = 32       # batch size per GPU

    # Number of GPUs available
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available for DDP.")

    # Launch distributed training
    mp.spawn(
        main,
        args=(world_size, total_epochs, save_every, batch_size),
        nprocs=world_size
    )
