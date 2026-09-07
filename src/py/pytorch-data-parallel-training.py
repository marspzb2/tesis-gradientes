#!/usr/bin/env python3
import runpod
import os
import sys
import tempfile
import traceback
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
REPORT_LOSS_EVERY = 1
PRINT_LOSS_EVERY = 300
EPOCHS = 10
BATCH_SIZE = 32

GPU_AVAILABLE = torch.cuda.is_available()
BACKEND = 'nccl' if GPU_AVAILABLE else 'gloo'
# BACKEND = 'gloo'
COMPRESSION_NAME = os.environ.get("COMPRESSION_SCHEME") or "none"
ERROR_FEEDBACK_ENABLED = os.environ.get("ERROR_FEEDBACK_ENABLED", "false").lower() == "true"
WARMUP_ENABLED = os.environ.get("WARMUP_ENABLED", "false").lower() == "true"
METHOD_PARAMS_NAMES = None

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
    POWER_SGD_RANK = int(os.environ.get("POWER_SGD_RANK", 5))
    POWER_SGD_ITERATIONS = int(os.environ.get("POWER_SGD_ITERATIONS", 1))
    POWER_SGD_PARAM_HANDLE = os.environ.get("POWER_SGD_PARAM_HANDLE", "by_param")
    COMPRESSION_CONFIG = POWER_SGD_PARAM_HANDLE, lambda tensor: compression_python.PowerSGD(POWER_SGD_RANK)
    METHOD_PARAMS_NAMES = f"r{POWER_SGD_RANK}i{POWER_SGD_ITERATIONS}p{POWER_SGD_PARAM_HANDLE}"
elif COMPRESSION_NAME.lower() == "none":
    COMPRESSION_CONFIG = "whole", lambda tensor: compression_python.NoCompression()
elif COMPRESSION_NAME.lower() == "qsgd":
    QSGD_Q = int(os.environ.get("QSGD_Q", 8))
    QSGD_DITHER = os.environ.get("QSGD_DITHERING_DISABLED", "false").lower() != "true"
    QSGD_BUCKET_SIZE = int(os.environ.get("QSGD_BUCKET_SIZE", 512))
    QSGD_NORM_P = int(os.environ.get("QSGD_NORM_P", 2)) if os.environ.get("QSGD_NORM_P", 2) != 'inf' else 10000000
    COMPRESSION_CONFIG = "whole", lambda tensor: compression_python.QSGD(QSGD_Q, QSGD_BUCKET_SIZE, QSGD_NORM_P,
                                                                         QSGD_DITHER)
    METHOD_PARAMS_NAMES = f"q{QSGD_Q}b{QSGD_BUCKET_SIZE}d{"T" if QSGD_DITHER else "F"}p{"inf" if QSGD_NORM_P > 10 else QSGD_NORM_P}"

else:
    raise RuntimeError("Error compression not supported")

OUTPUT_NAME = os.environ.get(
    "OUTPUT_NAME",
    f"{COMPRESSION_NAME}{"-ef" if ERROR_FEEDBACK_ENABLED else ""}{"-warmup" if WARMUP_ENABLED else ""}{f"-{METHOD_PARAMS_NAMES}" if METHOD_PARAMS_NAMES else ""}")

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


class VectorTracker:
    def __init__(self):
        self.vector_l2_2 = None
        self.vector_l1 = None
        self.vector_l0 = None
        self.vector_l_inf = None
        self.bytes = 0

    def track_vector_part(self, tensor):
        self.bytes += tensor.numel() * tensor.dtype.itemsize

        if tensor is not None and self.vector_l2_2 is not None:
            self.vector_l2_2 += torch.sum(tensor.square())
            self.vector_l1 += torch.sum(tensor.abs())
            self.vector_l0 += torch.count_nonzero(tensor)
            self.vector_l_inf = torch.maximum(self.vector_l_inf, torch.max(tensor.abs()))
        elif self.vector_l2_2 is None:
            self.vector_l2_2 = torch.sum(tensor.square())
            self.vector_l1 = torch.sum(tensor.abs())
            self.vector_l0 = torch.count_nonzero(tensor)
            self.vector_l_inf = torch.max(tensor.abs())

    def reset_vector_metrics(self):
        self.reset_vector_norms()
        self.bytes = 0

    def reset_vector_norms(self):
        self.vector_l2_2 = None
        self.vector_l1 = None
        self.vector_l0 = None
        self.vector_l_inf = None

    def reset_bytes(self):
        self.bytes = 0

    def add_summary_norms(self, writer, name_prefix, iteration):
        writer.add_scalar(f'{name_prefix}_norm_l2', torch.sqrt(self.vector_l2_2), iteration)
        writer.add_scalar(f'{name_prefix}_norm_l1', self.vector_l1, iteration)
        writer.add_scalar(f'{name_prefix}_norm_l0', self.vector_l0, iteration)
        writer.add_scalar(f'{name_prefix}_norm_linf', self.vector_l_inf, iteration)

    def add_relative_norm(self, relative_to_tensor_tracker, writer, name_prefix, iteration):
        writer.add_scalar(f'{name_prefix}_relative_l2',
                          torch.sqrt(self.vector_l2_2 / relative_to_tensor_tracker.vector_l2_2)
                          if relative_to_tensor_tracker.vector_l2_2 > 0 else -1, iteration)
        writer.add_scalar(f'{name_prefix}_relative_l1',
                          self.vector_l1 / relative_to_tensor_tracker.vector_l1
                          if relative_to_tensor_tracker.vector_l1 > 0 else -1, iteration)

    def add_summary_size_kb(self, writer, name_prefix, iteration):
        writer.add_scalar(f'{name_prefix}_size_kb', self.bytes / 1024., iteration)

    def add_summary_size_mb(self, writer, name_prefix, iteration):
        writer.add_scalar(f'{name_prefix}_size_mb', self.bytes / (1024. ** 2), iteration)

    def add_summary_size_gb(self, writer, name_prefix, iteration):
        writer.add_scalar(f'{name_prefix}_size_gb', self.bytes / (1024. ** 3), iteration)


class CompressionPythonState:
    def __init__(self):
        self.original_tracker = VectorTracker()
        self.reconstructed_tracker = VectorTracker()
        self.error_tracker = VectorTracker()
        self.g_mean_tracker = VectorTracker()
        self.trackers = [self.original_tracker, self.reconstructed_tracker, self.error_tracker, self.g_mean_tracker]
        self.reconstructed_dot_g_mean = 0

    def add_original_vector(self, tensor):
        self.original_tracker.track_vector_part(tensor)

    def add_reconstructed_vector(self, tensor):
        self.reconstructed_tracker.track_vector_part(tensor)

    def add_error_vector(self, tensor):
        self.error_tracker.track_vector_part(tensor)

    def add_g_mean_vector(self, tensor):
        self.g_mean_tracker.track_vector_part(tensor)

    def add_cdot_reconstructed_original(self, reconstructed, g_mean):
        self.reconstructed_dot_g_mean += torch.dot(reconstructed, g_mean)

    def get_original_bytes(self):
        return self.original_tracker.bytes

    def get_original_mb(self):
        return self.original_tracker.bytes / 1024 / 1024

    def reset_network_stats(self):
        for tracker in self.trackers:
            tracker.reset_bytes()

    def get_reconstruction_cosine_dist(self):
        if (not self.g_mean_tracker.vector_l2_2 or not self.reconstructed_tracker.vector_l2_2):
            return -100000
        return 1 - self.reconstructed_dot_g_mean / torch.sqrt(
            self.g_mean_tracker.vector_l2_2 * self.reconstructed_tracker.vector_l2_2)

    def get_reconstruction_angle(self):
        if (not self.g_mean_tracker.vector_l2_2 or not self.reconstructed_tracker.vector_l2_2):
            return -100000
        cos = self.reconstructed_dot_g_mean / torch.sqrt(
            self.g_mean_tracker.vector_l2_2 * self.reconstructed_tracker.vector_l2_2)
        if cos >= -100:
            return torch.rad2deg(torch.acos(torch.clamp(cos, -1, 1)))
        else:
            return -100000

    def reset_norms(self):
        self.reconstructed_dot_g_mean = 0
        for tracker in self.trackers:
            tracker.reset_vector_norms()

def add_mean_from_c(series_name,metric_name,writer,iteration):
    if compression_python.has_number_metric(metric_name):
        writer.add_scalar(series_name,compression_python.number_metric(metric_name).avg,iteration)
def add_min_from_c(series_name,metric_name,writer,iteration):
    if compression_python.has_number_metric(metric_name):
        writer.add_scalar(series_name,compression_python.number_metric(metric_name).min,iteration)
def add_max_from_c(series_name,metric_name,writer,iteration):
    if compression_python.has_number_metric(metric_name):
        writer.add_scalar(series_name,compression_python.number_metric(metric_name).max,iteration)
def add_vector_norms(series_name,metric_name,writer,iteration):
    if compression_python.has_vector_metric(metric_name):
        writer.add_scalar(f"{series_name}_norm_0",compression_python.vector_metric(metric_name).norm_0,iteration)
        writer.add_scalar(f"{series_name}_norm_1",compression_python.vector_metric(metric_name).norm_1,iteration)
        writer.add_scalar(f"{series_name}_norm_2",compression_python.vector_metric(metric_name).norm_2,iteration)
        writer.add_scalar(f"{series_name}_norm_inf",compression_python.vector_metric(metric_name).norm_inf,iteration)

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

        setup_node(rank, world_size)
        print(f"Node setted up: {rank}")
        with SummaryWriter(f"{OUT_FOLDER}/summary-{rank}/{OUTPUT_NAME}") as writer:
            iteration = 0
            compression_ctx = compression_python.CompressionFilterCtx(WORLD_SIZE)
            python_state = CompressionPythonState()
            compression_contexts_by_method = {}
            sampler = DistributedSampler(DS, num_replicas=world_size, rank=rank)
            print(f"Loading loaders... {rank}")
            dataloader = DataLoader(DS, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS,
                                    pin_memory=GPU_AVAILABLE)
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
                python_state.add_original_vector(gradient_tensor)

                param_config, compression_method_creator = COMPRESSION_CONFIG
                no_compression = False
                method_by_param = False  # True #Powersgd
                square_original_matrix = False
                get_reconstruction_metric = True
                if param_config == "by_param":
                    method_by_param = True

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
                    python_state.add_reconstructed_vector(reconstructed_tensor)
                    python_state.add_g_mean_vector(averaged_result_of_gradient_tensor)
                    python_state.add_error_vector(reconstructed_tensor - averaged_result_of_gradient_tensor)
                    python_state.add_cdot_reconstructed_original(reconstructed_tensor,
                                                                 averaged_result_of_gradient_tensor)
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
            validation_transmitted_kbytes = 0
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
                    python_state.reset_norms()
                    running_loss += loss.item()
                    report_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    validation_transmitted_kbytes += compression_python.get_bytes()/1024.
                    if iteration % REPORT_LOSS_EVERY == 0:
                        writer.add_scalar('loss', report_loss / REPORT_LOSS_EVERY, iteration)
                        python_state.error_tracker.add_summary_norms(writer, "reconstruction_error", iteration)
                        python_state.reconstructed_tracker.add_summary_norms(writer, "reconstruction", iteration)
                        python_state.original_tracker.add_summary_norms(writer, "node_g", iteration)
                        python_state.g_mean_tracker.add_summary_norms(writer, "reduced_g_mean", iteration)
                        python_state.reconstructed_tracker.add_relative_norm(python_state.g_mean_tracker,
                                                                             writer,
                                                                             "reconstruction_norm_over_no_compression_rate",
                                                                             iteration)
                        python_state.reconstructed_tracker.add_relative_norm(python_state.original_tracker,
                                                                             writer,
                                                                             "reconstruction_norm_over_original_rate",
                                                                             iteration)
                        python_state.error_tracker.add_relative_norm(python_state.g_mean_tracker,
                                                                     writer,
                                                                     "error_over_no_compression_rate",
                                                                     iteration)
                        writer.add_scalar('cosine_dist_reconstruction_g_mean',
                                          python_state.get_reconstruction_cosine_dist(), iteration)
                        writer.add_scalar('reconstruction_g_mean_angle', python_state.get_reconstruction_angle(),
                                          iteration)
                        python_state.g_mean_tracker.add_summary_size_kb(writer, "no_compression", iteration)
                        writer.add_scalar('sent_kb', compression_python.get_bytes() / 1024., iteration)
                        writer.add_scalar('compression_ratio',
                                          compression_python.get_bytes() / python_state.g_mean_tracker.bytes,
                                          iteration)
                        add_mean_from_c('average_run_length','zero-run-length',writer,iteration)
                        add_max_from_c('max_run_length','zero-run-length',writer,iteration)
                        add_vector_norms("error_feedback_vector","error_feed_back_vector",writer,iteration)

                        report_loss = 0
                        compression_python.reset_network_stats()
                        python_state.reset_norms()
                        python_state.reset_network_stats()
                        compression_python.reset_metrics(True)

                    writer.add_scalar("transmitted_kb_from_last_validation",validation_transmitted_kbytes,iteration)
                    writer.add_scalar("transmitted_mb_from_last_validation",validation_transmitted_kbytes/1024.,iteration)
                    if iteration % PRINT_LOSS_EVERY == 0:
                        print(f"Loss of {epoch_iteration}/{epoch} ({iteration})): {running_loss / PRINT_LOSS_EVERY}")
                        validate_model(ddp_model, validation_loader, criterion, writer, rank, iteration, epoch)
                        validation_transmitted_kbytes = 0
                        running_loss = 0

        print(f"Finished running basic DDP example on rank {rank}.")
    except Exception:
        print(f"Failed running basic DDP example on rank {rank}:")
        traceback.print_exc()
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        mp.spawn(train_model,
                 args=(WORLD_SIZE,),
                 nprocs=WORLD_SIZE,
                 join=True)
    except Exception:
        print("Failed to spawn:")
        traceback.print_exc()
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
