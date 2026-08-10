#!/usr/bin/env python3
import runpod
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from concurrent.futures import wait
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet101

from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torchvision import datasets, transforms, models as torch_models
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from math import ceil
import os
import random
from time import sleep
import warnings
import compression_python
from torch.utils.data.distributed import DistributedSampler

OUT_FOLDER = os.environ.get("OUT_FOLDER") or f"{os.getcwd()}/out/summary"
DATA_FOLDER = os.environ.get("DATA_FOLDER") or f"{os.getcwd()}/data"
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 2))
CPUS_AVAILABLE = 8
try:
    CPUS_AVAILABLE = os.os.sched_getaffinity(0)
except:
    CPUS_AVAILABLE = os.cpu_count() or 8
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(16, max(4, CPUS_AVAILABLE // WORLD_SIZE - 1))))
MODEL_NAME = "alexnet"
DATASET_NAME = "cifar10"
REPORT_LOSS_EVERY = 10
PRINT_LOSS_EVERY = 300
EPOCHS = 10
BATCH_SIZE = 32

GPU_AVAILABLE = torch.cuda.is_available()
BACKEND = 'nccl' if GPU_AVAILABLE else 'gloo'
# BACKEND = 'gloo'
COMPRESSION_NAME = os.environ.get("COMPRESSION_SCHEME") or "none"

OUTPUT_NAME=os.environ.get("OUTPUT_NAME",f"{COMPRESSION_NAME}")

seed = int(os.environ.get('SEED', 42))

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


warnings.filterwarnings(
    "ignore",
    category=np.exceptions.VisibleDeprecationWarning,
    module=r"torchvision\.datasets\.cifar",
)
def alex_net_creator():
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def create_model():
        alexnet = torchvision.models.alexnet(weights=None)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(alexnet.parameters(), lr=0.005, momentum=0.9)
        optimizer_params = {'momentum': 0.9, 'lr': 0.005}
        return alexnet, criterion, optimizer_params

    return transform, 'alexnet', create_model


COMPRESSION_CONFIG = None
if COMPRESSION_NAME.lower() == "powersgd":
    COMPRESSION_CONFIG = "by_param", lambda tensor: compression_python.PowerSGD(
        int(os.environ.get("POWER_SGD_RANK", 5)))
elif COMPRESSION_NAME.lower() == "none":
    COMPRESSION_CONFIG = "whole", lambda tensor: compression_python.NoCompression()
else:
    raise RuntimeError("Error compression not supported")

COMPRESSOR = None
# COMPRESSOR = qsgd_cpu_extension.qsgd_cpp_compress_hook

# if COMPRESSION_NAME == 'qsgd':
#    COMPRESSOR = QSGD
# elif COMPRESSION_NAME == 'no':
#    COMPRESSOR = NoCompression
# else:
#    raise Exception(f"Compression not found {COMPRESSOR}")
# COMPRESSOR = NoCompression
MODEL_INFO = None
if MODEL_NAME == "alexnet":
    MODEL_INFO = alex_net_creator
else:
    raise Exception(f"No model found {MODEL_NAME}")

transform, MODEL_NAME, MODEL_CREATOR = MODEL_INFO()

DS = None
VALIDATION_DS = None
if DATASET_NAME == "cifar10":
    DS = datasets.CIFAR10(root=DATA_FOLDER, train=True, download=True, transform=transform)
    VALIDATION_DS = datasets.CIFAR10(root=DATA_FOLDER, train=False, download=True, transform=transform)
else:
    raise Exception(f"No dataset found {DATASET_NAME}")

VALIDATION_BATCH = BATCH_SIZE


def setup_node(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # We want to be able to train our model on an `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
    # such as CUDA, MPS, MTIA, or XPU.
    # acc = torch.accelerator.current_accelerator()
    # backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize the process group
    dist.init_process_group(BACKEND, rank=rank, world_size=world_size)
    compression_python.initialize(dist.distributed_c10d._get_default_group())


def cleanup():
    dist.destroy_process_group()


def validate_model(model, validation_loader, loss_function, writer, rank, iteration, epoch):
    dist.barrier()
    if rank == 0:
        print(f"Validating dataset {iteration}/{epoch}...")

        total_loss = 0
        total_accuracy = 0

        model = model.module or model
        model.eval()
        with torch.no_grad():
            #  iterating through batches
            total_correct = 0
            total_samples = 0
            for images, labels in tqdm(validation_loader):
                if GPU_AVAILABLE:
                    images, labels = images.to(rank), labels.to(rank)
                classifications = model(images)
                total_loss += loss_function(classifications, labels).item()
                # Step 1: Get the predicted class (index of the max value)
                _, preds = torch.max(classifications, dim=1)

                # Step 2: Compare with labels and calculate accuracy
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            total_accuracy = total_correct / total_samples

            writer.add_scalar('validation_loss', total_loss, iteration)
            writer.add_scalar('validation_loss_mean', total_loss / len(validation_loader), iteration)
            writer.add_scalar('validation_accuracy', total_accuracy, iteration)
            print(
                f"Finished validating dataset on rank {rank}: {total_loss / len(validation_loader)} -- {total_accuracy * 100} %")
        model.train()
    else:
        print("Skipping validation")
    dist.barrier()


class CompressionPythonState:
    def __init__(self):
        self.original_bytes = 0
        self.reset_reconstruction_error()
        self.reset_reconstruction_power()

    def add_vector_size(self, vector):
        self.original_bytes += vector.numel() * vector.dtype.itemsize

    def get_original_bytes(self):
        return self.original_bytes

    def get_original_mb(self):
        return self.original_bytes / 1024 / 1024

    def reset_network_stats(self):
        self.original_bytes = 0

    def accumulate_reconstruction_error(self, reconstructed, original):
        error = reconstructed - original
        bucket_l2_squared = torch.sum(error.square())
        bucket_l1 = torch.sum(error.abs())
        bucket_inf = torch.max(error.abs())

        self.error_l2_squared = (
            bucket_l2_squared
            if self.error_l2_squared is None
            else self.error_l2_squared + bucket_l2_squared
        )
        self.error_l1 = (
            bucket_l1
            if self.error_l1 is None
            else self.error_l1 + bucket_l1
        )
        self.error_inf = (
            bucket_inf
            if self.error_inf is None
            else torch.maximum(self.error_inf, bucket_inf)
        )

    def get_reconstruction_error_inf(self):
        return self.error_inf

    def get_reconstruction_error_l2(self):
        if self.error_l2_squared is None:
            return None
        return torch.sqrt(self.error_l2_squared)

    def get_reconstruction_error_l1(self):
        return self.error_l1

    def reset_reconstruction_error(self):
        self.error_l2_squared = None
        self.error_l1 = None
        self.error_inf = None

    def get_reconstruction_norm_l2(self):
        if self.reconstruction_norm_l2_squared is None:
            return None
        return torch.sqrt(self.reconstruction_norm_l2_squared)

    def get_reconstruction_norm_l1(self):
        return self.reconstruction_norm_l1

    def get_reconstruction_norm_l0(self):
        return self.reconstruction_norm_l0

    def get_reconstruction_norm_inf(self):
        return self.reconstruction_norm_inf

    def accumulate_reconstruction_power(self, reconstructed):
        bucket_l2_squared = torch.sum(reconstructed.square())
        bucket_l1 = torch.sum(reconstructed.abs())
        bucket_l0 = torch.count_nonzero(reconstructed)
        bucket_inf = torch.max(reconstructed.abs())

        self.reconstruction_norm_l2_squared = (
            bucket_l2_squared
            if self.reconstruction_norm_l2_squared is None
            else self.reconstruction_norm_l2_squared + bucket_l2_squared
        )
        self.reconstruction_norm_l1 = (
            bucket_l1
            if self.reconstruction_norm_l1 is None
            else self.reconstruction_norm_l1 + bucket_l1
        )
        self.reconstruction_norm_l0 = (
            bucket_l0
            if self.reconstruction_norm_l0 is None
            else self.reconstruction_norm_l0 + bucket_l0
        )
        self.reconstruction_norm_inf = (
            bucket_inf
            if self.reconstruction_norm_inf is None
            else torch.maximum(self.reconstruction_norm_inf, bucket_inf)
        )

    def reset_reconstruction_power(self):
        self.reconstruction_norm_l2_squared = None
        self.reconstruction_norm_l1 = None
        self.reconstruction_norm_l0 = None
        self.reconstruction_norm_inf = None


def train_model(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    if GPU_AVAILABLE:
        available_gpus = torch.cuda.device_count()
        if world_size > available_gpus:
            raise RuntimeError(
                f"WORLD_SIZE={world_size}, pero solo hay "
                f"{available_gpus} GPUs CUDA disponibles"
            )
        torch.cuda.set_device(rank)

    try:
        # node_compression = COMPRESSOR()

        setup_node(rank, world_size)
        print(f"Node setted up: {rank}")
        with SummaryWriter(f"{OUT_FOLDER}/summary-{rank}/{OUTPUT_NAME}") as writer:
            iteration = 0
            compression_ctx = compression_python.CompressionFilterCtx(WORLD_SIZE)
            python_state = CompressionPythonState()
            compression_contexts_by_method = {}
            sampler = DistributedSampler(DS, num_replicas=world_size, rank=rank)
            print(f"Loading loaders... {rank}")
            dataloader = DataLoader(DS, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=GPU_AVAILABLE)
            validation_loader = DataLoader(VALIDATION_DS, batch_size=VALIDATION_BATCH, num_workers=NUM_WORKERS,
                                           pin_memory=GPU_AVAILABLE) if rank == 0 else None
            print(f"Loaders loaded... {rank}")
            _model, criterion, optimizer_params = MODEL_CREATOR()
            if GPU_AVAILABLE:
                _model.to(rank)
                ddp_model = DDP(_model, device_ids=[rank])
            else:
                ddp_model = DDP(_model)

            # compression_state = node_compression.create_state()

            def compress_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
                # print(
                #     f"rank={dist.get_rank()} "
                #     f"iteration={iteration} "
                #     f"bucket={bucket.index()} "
                #     f"last={bucket.is_last()} "
                #     f"elements={bucket.buffer().numel()}"
                # )
                gradient_tensor = bucket.buffer()
                param_config, compression_method_creator = COMPRESSION_CONFIG
                no_compression = False
                method_by_param = False  # True #Powersgd
                square_original_matrix = False
                get_reconstruction_metric = True
                if param_config == "by_param":
                    method_by_param = True

                python_state.add_vector_size(gradient_tensor)
                world_size = dist.get_world_size()
                fut = torch.futures.Future()
                tensor_copy = gradient_tensor.clone()
                if no_compression:
                    fut.set_result(average_tensor_on_the_network(gradient_tensor))
                elif method_by_param:
                    current_index_in_buffer = 0
                    for param in bucket.parameters():
                        original_shape = param.shape
                        total_elements = param.numel()
                        param_key = param.data_ptr()
                        if param_key not in compression_contexts_by_method:
                            compression_contexts_by_method[param_key] = compression_python.CompressionFilterCtx(
                                WORLD_SIZE)
                            print(f"Initialized context for powersgd on param: {param.data_ptr()}")
                        tensor_elements = gradient_tensor[
                            current_index_in_buffer:current_index_in_buffer + total_elements]
                        matrix_to_compress = tensor_elements.view(original_shape)
                        if matrix_to_compress.dim() == 4:
                            matrix_to_compress = matrix_to_compress.view(original_shape[0], -1)

                        current_index_in_buffer += total_elements
                        method = compression_method_creator(matrix_to_compress)
                        result = method.run(matrix_to_compress, compression_contexts_by_method[param_key])
                        # print("Paso")
                        tensor_elements.copy_(result.view(-1))
                        # print("Termino la copia")
                    fut.set_result(gradient_tensor)
                elif square_original_matrix:
                    print("Not done yet")
                else:
                    result = compression_method_creator(gradient_tensor).run(gradient_tensor, compression_ctx)
                    fut.set_result(result)
                if get_reconstruction_metric:
                    # print("Reconstruction metric computing")
                    averaged_result_of_gradient_tensor = average_tensor_on_the_network(tensor_copy)
                    reconstructed_tensor = fut.wait()
                    python_state.accumulate_reconstruction_error(reconstructed_tensor,
                                                                 averaged_result_of_gradient_tensor)
                    python_state.accumulate_reconstruction_power(reconstructed_tensor)
                    # print("Reconstruction metric computed")
                    # print(reconstructed_tensor[:10])
                    # print(averaged_result_of_gradient_tensor[:10])

                return fut

            def average_tensor_on_the_network(gradient_tensor):
                averaged_tensor = gradient_tensor
                dist.all_reduce(
                    averaged_tensor,
                    op=dist.ReduceOp.SUM,
                )
                return averaged_tensor.div_(dist.get_world_size())

            ddp_model.register_comm_hook(None, compress_hook)
            # ddp_model.register_comm_hook(None,qsgd_hook)
            optimizer = optim.SGD(ddp_model.parameters(), **optimizer_params)
            running_loss = 0
            report_loss = 0
            print(f"Model in card... {rank}")
            ddp_model.train()
            for epoch in range(EPOCHS):
                sampler.set_epoch(epoch)
                for epoch_iteration, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
                    if GPU_AVAILABLE:
                        data, target = data.to(rank), target.to(rank)
                    # print(f"Hola {rank}/{epoch}/{iteration}")
                    iteration += 1
                    optimizer.zero_grad()
                    outputs = ddp_model(data)
                    loss = criterion(outputs, target)
                    python_state.reset_reconstruction_error()
                    python_state.reset_reconstruction_power()
                    running_loss += loss.item()
                    report_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                    if iteration % REPORT_LOSS_EVERY == 0:
                        writer.add_scalar('loss', report_loss / REPORT_LOSS_EVERY, iteration)
                        writer.add_scalar('transferred_kb', compression_python.get_bytes() / 1024., iteration)
                        writer.add_scalar('original_kb', python_state.get_original_bytes() / 1024., iteration)
                        if (python_state.get_reconstruction_norm_l2() is not None):
                            writer.add_scalar('reconstruction_norm_l2', python_state.get_reconstruction_norm_l2(),
                                              iteration)
                            writer.add_scalar('reconstruction_norm_l1', python_state.get_reconstruction_norm_l1(),
                                              iteration)
                            writer.add_scalar('reconstruction_norm_l0', python_state.get_reconstruction_norm_l0(),
                                              iteration)
                            writer.add_scalar('reconstruction_norm_inf', python_state.get_reconstruction_norm_inf(),
                                              iteration)
                        if python_state.get_reconstruction_error_l2() is not None:
                            writer.add_scalar('reconstruction_error_l2', python_state.get_reconstruction_error_l2(),
                                              iteration)
                            writer.add_scalar('reconstruction_error_l1', python_state.get_reconstruction_error_l1(),
                                              iteration)

                        # print(
                        #     f"Megabytes {compression_python.get_bytes()}/{python_state.get_original_bytes()}({compression_python.get_bytes() / python_state.get_original_bytes()})")
                        # print(f"Model loss {report_loss / REPORT_LOSS_EVERY})")
                        report_loss = 0
                        compression_python.reset_network_stats()
                        python_state.reset_network_stats()

                    if iteration % PRINT_LOSS_EVERY == 0:
                        print(f"Loss of {epoch_iteration}/{epoch} ({iteration})): {running_loss / PRINT_LOSS_EVERY}")
                        validate_model(ddp_model, validation_loader, criterion, writer, rank, iteration, epoch)
                        running_loss = 0

        print(f"Finished running basic DDP example on rank {rank}.")
    except Exception as e:
        print(f"Failed running basic DDP example on rank {rank}: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        mp.spawn(train_model,
                 args=(WORLD_SIZE,),
                 nprocs=WORLD_SIZE,
                 join=True)
    except:
        print("Failed to spawn")
    finally:
        try:
            pod_id = os.environ["RUNPOD_POD_ID"]
            api_key = os.environ["RUNPOD_API_KEY"]
            if pod_id and api_key:
                runpod.api_key = api_key
                runpod.stop_pod(pod_id)
            else:
                print("Runpod env not detected")
        finally:
            sleep(60)
            print("Ended")
