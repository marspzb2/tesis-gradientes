//
// Created by Marcos Perez on 4/6/26.
//
module;
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <torch/csrc/distributed/c10d/Types.hpp>

export module Network;

export enum class Topology {
    STAR,
    RING
};

void my_function(
    const c10::intrusive_ptr<c10d::ProcessGroup> &pg
) {
    auto rank = pg->getRank();
    auto size = pg->getSize();

    std::cout << "rank=" << rank
            << " size=" << size << std::endl;
}

export class Network {
private:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group;
    std::optional<Topology> topology;
    uint64_t bytes = 0;

    Network() = default;

public:
    Network(const Network &) = delete;

    Network &operator=(const Network &) = delete;

    static Network &instance() {
        static Network network;
        return network;
    }

    uint64_t get_bytes() {
        return this->bytes;
    }

    void reset_network_stats() {
        this->bytes = 0;
    }

    void initialize(
        c10::intrusive_ptr<c10d::ProcessGroup> pg,
        Topology t
    ) {
        if (process_group) {
            throw std::runtime_error("Network already initialized");
        }

        process_group = std::move(pg);
        topology = std::move(t);
        std::cout << "Network initialized" << std::endl;
    }

    c10::intrusive_ptr<c10d::ProcessGroup> get_process_group() {
        return process_group;
    }

    c10::intrusive_ptr<c10d::Work> _all_gather(std::vector<torch::Tensor> &inputs,
                                               std::vector<std::vector<torch::Tensor> > &outputs) {
        return this->process_group->allgather(outputs, inputs, c10d::AllgatherOptions());
    }

    std::vector<std::vector<torch::Tensor> > make_send_copies_same_size(std::vector<torch::Tensor> send_tensors,
                                                                        int world_size) {
        std::vector<std::vector<torch::Tensor> > all_tensors(send_tensors.size());
        int send_idx = 0;
        for (const torch::Tensor &send_tensor: send_tensors) {
            all_tensors[send_idx] = std::vector<torch::Tensor>();
            // transfer_list[send_idx] = send_tensor;
            for (int j = 0; j < world_size; j++) {
                all_tensors[send_idx].push_back(torch::empty_like(send_tensor));
            }
            send_idx++;
        }
        return all_tensors;
    }

    template<typename... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<torch::Tensor> > > all_gather_same_size(
        Args &&... tensors) {
        std::vector<torch::Tensor> sizes_tensors{torch::tensor(tensors.sizes(), torch::kLong)...};
        int world_size = process_group->getSize();

        std::vector<std::vector<torch::Tensor> > all_tensors = make_send_copies_same_size(sizes_tensors, world_size);
        this->add_bytes(sizes_tensors);
        return std::make_tuple(this->_all_gather(sizes_tensors, all_tensors), all_tensors);
    }

    template<typename... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<torch::Tensor> > > all_gather(
        bool same_size = true, Args &&... args) {
        if (same_size) {
            return all_gather_same_size(std::forward<Args>(args)...);
        } else {
            std::vector<torch::Tensor> send_tensors{std::forward<Args>(args)...};
            if (send_tensors.size() != 1) {
                throw std::invalid_argument(
                    "Variable-size all_gather supports one tensor per call");
            }

            torch::Tensor send_tensor = send_tensors.front();
            if (send_tensor.dim() != 1) {
                throw std::invalid_argument(
                    "Variable-size all_gather expects a 1D tensor");
            }

            int world_size = process_group->getSize();

            auto [size_work, gathered_sizes] =
                this->all_gather_sizes(send_tensor);
            size_work->wait();

            std::vector<int64_t> sizes;
            sizes.reserve(world_size);
            int64_t max_size = 0;
            for (const torch::Tensor &size_tensor: gathered_sizes.front()) {
                int64_t size = size_tensor[0].item<int64_t>();
                sizes.push_back(size);
                max_size = std::max(max_size, size);
            }

            torch::Tensor padded = torch::zeros({max_size}, send_tensor.options());
            padded.slice(0, 0, send_tensor.numel()).copy_(send_tensor);

            std::vector<torch::Tensor> padded_inputs{padded};
            auto gathered_padded =
                make_send_copies_same_size(padded_inputs, world_size);
            auto work = this->_all_gather(padded_inputs, gathered_padded);
            work->wait();

            for (int rank = 0; rank < world_size; ++rank) {
                gathered_padded[0][rank] =
                    gathered_padded[0][rank].slice(0, 0, sizes[rank]);
            }

            this->add_bytes(padded_inputs);
            return std::make_tuple(work, std::move(gathered_padded));
        }
    }

    template<typename... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<torch::Tensor> > > all_gather_sizes(
        Args &&... tensors) {
        std::vector<torch::Tensor> sizes_tensors{
            torch::tensor(tensors.sizes(), torch::TensorOptions().device(tensors.device()).dtype(torch::kLong))...
        };
        this->add_bytes(sizes_tensors);
        int world_size = process_group->getSize();
        std::vector<std::vector<torch::Tensor> > all_tensors = make_send_copies_same_size(sizes_tensors, world_size);
        return std::make_tuple(this->_all_gather(sizes_tensors, all_tensors), all_tensors);
    }

    template<typename... Args>
    c10::intrusive_ptr<c10d::Work> all_reduce(Args &&... tensors) {
        return this->all_reduce_with_op(
            c10d::ReduceOp::SUM,
            std::forward<Args>(tensors)...
        );
    }

    template<typename... Args>
    c10::intrusive_ptr<c10d::Work> all_reduce_with_op(
        c10d::ReduceOp::RedOpType operation,
        Args &&... tensors) {
        std::vector<torch::Tensor> send_tensors{std::forward<Args>(tensors)...};
        this->add_bytes(send_tensors);

        c10d::AllreduceOptions options;
        options.reduceOp = operation;
        return this->process_group->allreduce(send_tensors, options);
    }

    void add_bytes(const std::vector<std::vector<torch::Tensor> > &send_tensors) {
        for (const auto &tensor: send_tensors) {
            this->add_bytes(tensor);
        }
    }

    void add_bytes(const std::vector<torch::Tensor> &send_tensors) {
        uint64_t total_bytes = 0;
        for (const auto &tensor: send_tensors) {
            total_bytes += tensor.numel() * tensor.element_size();
        }
        this->bytes += total_bytes;
    }

    /*
        c10d::Work send_to_master() {


        }

        c10d::Work recv_in_master() {
        }

        c10d::Work send_from_master() {
        }

        c10d::Work recv_from_master() {
        }*/
};
