#!/usr/bin/env python3
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet101

from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchvision import datasets, transforms,models as torch_models
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm

from torch.utils.data.distributed import DistributedSampler

WORLD_SIZE= os.environ.get('WORLD_SIZE') or 2
MODEL_NAME = "alexnet"
DATASET_NAME = "cifar10"
REPORT_LOSS_EVERY=20
PRINT_LOSS_EVERY=300
EPOCHS=10
BATCH_SIZE=32

GPU_AVAILABLE=torch.cuda.is_available()
BACKEND= 'nccl' if GPU_AVAILABLE else 'gloo'

COMPRESSION_NAME = "qsgd"

seed=None
class DefaultState:
    def __init__(self,group=None):
        self.group=group
        self.total_bytes=0
        self.original_bytes=0
    def add_original_tensor(self,tensor):
        self.original_bytes = tensor.numel() * tensor.element_size()
    def add_tensor_bytes(self,tensor):
        self.total_bytes = tensor.numel() * tensor.element_size()
    def get_mb(self):
        return self.total_bytes/(1024**2)
    def get_original_mb(self):
        return self.original_bytes/(1024**2)

class Compression:
    def create_state(self,group=None):
        return DefaultState(group)

class QSGDCompression(Compression):
    def __init__(self):
        pass
    def compress(self,state,bucket:dist.GradBucket):
        tensor = bucket.buffer()
        state.add_original_tensor(tensor)
        state.add_tensor_bytes(tensor)
        return dist.all_reduce(tensor,group=state.group,async_op=True).get_future()
    def decompress(self,state,compressed_value):
        return compressed_value[0]

class NoCompression(Compression):
    def __init__(self):
        pass
    def compress(self,state,bucket:dist.GradBucket):
        tensor = bucket.buffer()
        return dist.all_reduce(tensor,group=state.group,async_op=True).get_future()
    def decompress(self,state,compressed_value):
        return compressed_value[0]




def alex_net_creator():

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def create_model():
            alexnet=torchvision.models.alexnet(weights=None)
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.SGD(alexnet.parameters(), lr=0.005, momentum=0.9)
            optimizer_params={'momentum':0.9, 'lr':0.005}
            return alexnet, criterion, optimizer_params

    return transform,'alexnet',create_model

COMPRESSOR=None
match COMPRESSION_NAME:
    case 'qsgd':
        COMPRESSOR = QSGDCompression
    case 'no':
        COMPRESSOR = NoCompression
    case _:
        raise Exception(f"Compression not found {COMPRESSOR}")




MODEL_INFO=None
if MODEL_NAME=="alexnet":
     MODEL_INFO= alex_net_creator
else:
    raise Exception(f"No model found {MODEL_NAME}")


transform,MODEL_NAME,MODEL_CREATOR= MODEL_INFO()

DS=None
VALIDATION_DS = None
if DATASET_NAME=="cifar10":
    DS=datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    VALIDATION_DS=datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
else:
    raise Exception(f"No dataset found {DATASET_NAME}")

VALIDATION_BATCH=BATCH_SIZE

def setup_node(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    #acc = torch.accelerator.current_accelerator()
    #backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def validate_model(model,validation_loader,loss_function,writer,rank):
    if rank!=0:
        return
    print("Validating dataset...")

    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        #  iterating through batches
        for images, labels in tqdm(validation_loader):
            if GPU_AVAILABLE:
                images, labels = images.to(rank), labels.to(rank)
            classifications = model(images)
            total_loss += loss_function(classifications, labels).item()
            # Step 1: Get the predicted class (index of the max value)
            _, preds = torch.max(classifications, dim=1)

            # Step 2: Compare with labels and calculate accuracy
            correct = (preds == labels).sum().item()
            accuracy = correct / labels.size(0)
            total_accuracy+=accuracy

        writer.add_scalar('validation_loss', total_loss, rank)
        writer.add_scalar('validation_loss_mean', total_loss / len(validation_loader), rank)
        writer.add_scalar('validation_accuracy', total_accuracy/len(validation_loader), rank)
        print(f"Finished validating dataset on rank {rank}: {total_loss/len(validation_loader)} -- {total_accuracy*100/len(validation_loader)} %")

def train_model(rank, world_size):

    print(f"Running basic DDP example on rank {rank}.")
    try:
        node_compression = COMPRESSOR()

        setup_node(rank, world_size)
        print(f"Node setted up: {rank}")
        writer=SummaryWriter(f"{os.getcwd()}/summary-{rank}")
        iteration=0
        sampler = DistributedSampler(DS, num_replicas=world_size, rank=rank)
        print(f"Loading loaders... {rank}")
        dataloader = DataLoader(DS, batch_size=BATCH_SIZE, sampler=sampler,num_workers=4,pin_memory=GPU_AVAILABLE)
        validation_loader = DataLoader(VALIDATION_DS, batch_size=VALIDATION_BATCH, num_workers=4,
                                       pin_memory=GPU_AVAILABLE) if rank==0 else None
        print(f"Loaders loaded... {rank}")
        _model, criterion, optimizer_params = MODEL_CREATOR()
        if GPU_AVAILABLE:
            _model.to(rank)
            ddp_model = DDP(_model, device_ids=[rank])
        else:
            ddp_model = DDP(_model)
        def compress_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            future = node_compression.compress(state,bucket)
            return future.then(lambda fut : node_compression.decompress(state,fut.value()))

        compression_state=node_compression.create_state()
        ddp_model.register_comm_hook(compression_state, compress_hook)
        optimizer = optim.SGD(ddp_model.parameters(), **optimizer_params)
        running_loss = 0
        report_loss=0
        print(f"Model in card... {rank}")
        ddp_model.train()
        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)
            for epoch_iteration,(data, target) in enumerate(dataloader):
                if GPU_AVAILABLE:
                    data, target = data.to(rank), target.to(rank)
                print(f"Hola {rank}/{epoch}/{iteration}")
                iteration+=1
                optimizer.zero_grad()
                outputs = ddp_model(data)
                loss = criterion(outputs, target)
                running_loss+=loss.item()
                report_loss += loss.item()
                loss.backward()
                optimizer.step()
                if iteration % REPORT_LOSS_EVERY == 0:
                    report_loss = 0
                    writer.add_scalar('loss', report_loss / REPORT_LOSS_EVERY, iteration)
                    print(f"Model loss {report_loss / REPORT_LOSS_EVERY})")
                    print(f"Megabytes {compression_state.get_mb()}/{compression_state.get_original_mb()}({compression_state.get_mb()/compression_state.get_original_mb()})")

                if iteration % PRINT_LOSS_EVERY == 0:
                    # if GPU_AVAILABLE:
                    #    dist.barrier(device_ids=[rank])

                    print(f"Loss of {iteration}/{epoch}: {running_loss / 1000.}")
                    validate_model(ddp_model,validation_loader, criterion, writer, rank)
                    running_loss = 0
                    # if GPU_AVAILABLE:
                    #    dist.barrier(device_ids=[rank])

        cleanup()
        print(f"Finished running basic DDP example on rank {rank}.")
    except Exception as e:
        print(f"Failed running basic DDP example on rank {rank}: {e}")

if __name__ == "__main__":
    mp.spawn(train_model,
             args=(WORLD_SIZE,),
             nprocs=WORLD_SIZE,
             join=True)
