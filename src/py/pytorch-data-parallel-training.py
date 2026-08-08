#!/usr/bin/env python3
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


# print(torch.__version__)
import compression_python
from torch.utils.data.distributed import DistributedSampler

OUT_FOLDER = os.environ.get("OUT_FOLDER") or "./out/summary"
WORLD_SIZE = os.environ.get('WORLD_SIZE') or 2
MODEL_NAME = "alexnet"
DATASET_NAME = "cifar10"
REPORT_LOSS_EVERY = 20
PRINT_LOSS_EVERY = 300
EPOCHS = 10
BATCH_SIZE = 32

GPU_AVAILABLE = torch.cuda.is_available()
BACKEND = 'nccl' if GPU_AVAILABLE else 'gloo'
BACKEND = 'gloo'
COMPRESSION_NAME = "qsgd"

seed = int(os.environ.get('SEED')) if os.environ.get('SEED') else 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


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
    DS = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    VALIDATION_DS = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
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


def validate_model(model, validation_loader, loss_function, writer, rank, iteration):
    if rank != 0:
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
            total_accuracy += accuracy

        writer.add_scalar('validation_loss', total_loss, iteration)
        writer.add_scalar('validation_loss_mean', total_loss / len(validation_loader), iteration)
        writer.add_scalar('validation_accuracy', total_accuracy / len(validation_loader), iteration)
        print(
            f"Finished validating dataset on rank {rank}: {total_loss / len(validation_loader)} -- {total_accuracy * 100 / len(validation_loader)} %")


class CompressionPythonState:
    def __init__(self):
        self.original_bytes = 0

    def add_vector_size(self, vector):
        self.original_bytes += vector.numel() * vector.dtype.itemsize

    def get_original_bytes(self):
        return self.original_bytes

    def get_original_mb(self):
        return self.original_bytes / 1024 / 1024

    def reset_network_stats(self):
        self.original_bytes = 0

    def set_reconstruction_error(self, reconstructed, original):
        error = reconstructed - original
        self.error_inf = torch.linalg.vector_norm(error, ord=float('inf'))
        self.error_l2 = torch.linalg.vector_norm(error, ord=2)
        self.error_l1 = torch.linalg.vector_norm(error, ord=1)

    def get_reconstruction_error_inf(self):
        return self.error_inf

    def get_reconstruction_error_l2(self):
        return self.error_l2

    def get_reconstruction_error_l1(self):
        return self.error_l1

    def reset_reconstruction_error(self):
        self.error_l2 = None
        self.error_l1 = None
        self.error_inf = None

    def get_reconstruction_norm_l2(self):
        return self.reconstruction_norm_l2

    def get_reconstruction_norm_l1(self):
        return self.reconstruction_norm_l1

    def get_reconstruction_norm_l0(self):
        return self.reconstruction_norm_l0

    def get_reconstruction_norm_inf(self):
        return self.reconstruction_norm_inf

    def set_reconstruction_power(self, reconstructed):
        self.reconstruction_norm_l2 = torch.linalg.vector_norm(reconstructed, ord=2)
        self.reconstruction_norm_l1 = torch.linalg.vector_norm(reconstructed, ord=1)
        self.reconstruction_norm_inf = torch.linalg.vector_norm(reconstructed, ord=float('inf'))
        self.reconstruction_norm_l0 = torch.linalg.vector_norm(reconstructed, ord=0)


def train_model(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")

    try:
        # node_compression = COMPRESSOR()

        setup_node(rank, world_size)
        print(f"Node setted up: {rank}")
        writer = SummaryWriter(f"{os.getcwd()}/summary-{rank}")
        iteration = 0
        sampler = DistributedSampler(DS, num_replicas=world_size, rank=rank)
        print(f"Loading loaders... {rank}")
        dataloader = DataLoader(DS, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=GPU_AVAILABLE)
        validation_loader = DataLoader(VALIDATION_DS, batch_size=VALIDATION_BATCH, num_workers=4,
                                       pin_memory=GPU_AVAILABLE) if rank == 0 else None
        print(f"Loaders loaded... {rank}")
        _model, criterion, optimizer_params = MODEL_CREATOR()
        if GPU_AVAILABLE:
            _model.to(rank)
            ddp_model = DDP(_model, device_ids=[rank])
        else:
            ddp_model = DDP(_model)

        def obtener_vectores_comprimidos(local_vector):
            """
            Usamos al gather para recuperar todos los gradientes comprimidos, soporta largo variable por lo que hay que pedir los sizes primero.
            :param local_vector:
            :return:
            """
            world_size = dist.get_world_size()
            local_compressed_gradient_size = torch.tensor([local_vector.numel()], dtype=torch.long,
                                                          device=local_vector.device)
            world_gradient_sizes = [torch.zeros(1, dtype=torch.long, device=local_vector.device) for _ in
                                    range(world_size)]
            dist.all_gather(world_gradient_sizes, local_compressed_gradient_size)
            # Aca estan los tamaños de los comprimidos
            gradient_sizes = [s.item() for s in world_gradient_sizes]
            print("Size of gradient vectors: ", gradient_sizes)

            # ahora si podemos pedir los gradientes
            gradientes_de_los_servidores = [
                torch.zeros(size, dtype=local_vector.dtype, device=local_vector.device)
                for size in gradient_sizes
            ]

            return dist.all_gather(gradientes_de_los_servidores, local_vector,
                                   async_op=True).get_future(), gradientes_de_los_servidores

        # compression_state = node_compression.create_state()
        python_state = CompressionPythonState()

        def compress_hook(state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            gradient_tensor = bucket.buffer()
            no_compression = False
            method_by_param = False  # True #Powersgd
            get_reconstruction_metric = True
            square_original_matrix = False

            compression_method_creator = lambda tensor: compression_python.PowerSGD(5)
            compression_method_creator = lambda tensor: compression_python.NoCompression()
            python_state.add_vector_size(gradient_tensor)
            world_size = dist.get_world_size()
            fut = torch.futures.Future()
            if no_compression:
                fut.set_result(average_tensor_on_the_network(gradient_tensor))
            elif method_by_param:
                current_index_in_buffer = 0
                for param in bucket.parameters():
                    original_shape = param.shape
                    total_elements = param.numel()
                    tensor_elements = gradient_tensor[current_index_in_buffer:current_index_in_buffer + total_elements]
                    matrix_to_compress = tensor_elements.view(original_shape)
                    if matrix_to_compress.dim() == 4:
                        matrix_to_compress = matrix_to_compress.view(original_shape[0], -1)

                    current_index_in_buffer += total_elements
                    method = compression_method_creator(matrix_to_compress)
                    compression_ctx = compression_python.CompressionFilterCtx(WORLD_SIZE)
                    result = method.run(matrix_to_compress, compression_ctx)
                    # print("Paso")
                    tensor_elements.copy_(result.view(-1))
                    # print("Termino la copia")
                fut.set_result(gradient_tensor)
            elif square_original_matrix:
                print("Not done yet")
            else:
                compression_ctx = compression_python.CompressionFilterCtx(WORLD_SIZE)
                result = compression_method_creator(gradient_tensor).run(gradient_tensor, compression_ctx)
                fut.set_result(result)
            if  get_reconstruction_metric:
                #print("Reconstruction metric computing")
                averaged_result_of_gradient_tensor = average_tensor_on_the_network(gradient_tensor)
                reconstructed_tensor = fut.wait()
                python_state.set_reconstruction_error(reconstructed_tensor, averaged_result_of_gradient_tensor)
                python_state.set_reconstruction_power(reconstructed_tensor)
                #print("Reconstruction metric computed")
                #print(reconstructed_tensor[:10])
                #print(averaged_result_of_gradient_tensor[:10])

            return fut

        def average_tensor_on_the_network(gradient_tensor) -> Any:
            averaged_tensor = gradient_tensor.clone()
            dist.all_reduce(
                averaged_tensor,
                op=dist.ReduceOp.SUM,
            )
            return averaged_tensor.div_(dist.get_world_size())

        def qsgd_hook(state, bucket):

            tensor = bucket.buffer()

            # =========================
            # COMPRESIÓN C++
            # =========================

            quantized, scale = qsgd_cpu_extension.compress_qsgd(tensor)

            # =========================
            # ACA normalmente harías:
            #
            # all_gather / all_reduce
            # sobre quantized y scale
            # =========================

            # =========================
            # DESCOMPRESIÓN C++
            # =========================

            decompressed = qsgd_cpu_extension.decompress_qsgd(
                quantized,
                scale
            )

            fut = torch.futures.Future()

            fut.set_result(decompressed)

            return fut

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
                running_loss += loss.item()
                report_loss += loss.item()
                loss.backward()
                optimizer.step()

                if iteration % REPORT_LOSS_EVERY == 0:
                    writer.add_scalar('loss', report_loss / REPORT_LOSS_EVERY, iteration)
                    writer.add_scalar('transferred_kb', compression_python.get_bytes() / 1024., iteration)
                    writer.add_scalar('original_kb', python_state.get_original_bytes() / 1024., iteration)
                    if(python_state.get_reconstruction_norm_l2() is not None):
                        writer.add_scalar('reconstruction_norm_l2', python_state.get_reconstruction_norm_l2(), iteration)
                        writer.add_scalar('reconstruction_norm_l1', python_state.get_reconstruction_norm_l1(), iteration)
                        writer.add_scalar('reconstruction_norm_l0', python_state.get_reconstruction_norm_l0(), iteration)
                        writer.add_scalar('reconstruction_norm_inf', python_state.get_reconstruction_norm_inf(), iteration)
                    if python_state.get_reconstruction_error_l2() is not None:
                        writer.add_scalar('reconstruction_error_l2', python_state.get_reconstruction_error_l2(), iteration)
                        writer.add_scalar('reconstruction_error_l1', python_state.get_reconstruction_error_l1(), iteration)

                    print(f"Model loss {report_loss / REPORT_LOSS_EVERY})")
                    print(
                        f"Megabytes {compression_python.get_bytes()}/{python_state.get_original_bytes()}({compression_python.get_bytes() / python_state.get_original_bytes()})")
                    report_loss = 0
                    compression_python.reset_network_stats()
                    python_state.reset_network_stats()

                if iteration % PRINT_LOSS_EVERY == 0:
                    # if GPU_AVAILABLE:
                    #    dist.barrier(device_ids=[rank])

                    print(f"Loss of {iteration}/{epoch}: {running_loss / PRINT_LOSS_EVERY}")
                    validate_model(ddp_model, validation_loader, criterion, writer, rank, iteration)
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
