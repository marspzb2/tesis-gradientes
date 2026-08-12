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
            std::vector<std::vector<torch::Tensor> > all_tensors(send_tensors.size());
            int world_size = process_group->getSize();

            auto [work,result_tensors] = this->all_gather_sizes(std::forward<Args>(args)...);
            work->wait();
            int send_idx = 0;

            for (const std::vector<torch::Tensor> &received_sizes: result_tensors) {
                auto tensor_options = send_tensors[send_idx].options();
                all_tensors[send_idx] = std::vector<torch::Tensor>();
                all_tensors[send_idx].reserve(world_size);
                for (const torch::Tensor &received_size: received_sizes) {
                    torch::Tensor cpu_size_tensor = received_size.to(torch::kCPU);
                    auto sizes_accessor = cpu_size_tensor.accessor<int64_t, 1>();
                    std::vector<int64_t> shape;
                    shape.reserve(sizes_accessor.size(0));
                    for (int64_t k = 0; k < sizes_accessor.size(0); k++) {
                        shape.push_back(sizes_accessor[k]);
                    }
                    all_tensors[send_idx].push_back(torch::empty(shape, tensor_options));
                }
                send_idx++;
            }
            this->add_bytes(send_tensors);
            return std::make_tuple(this->_all_gather(send_tensors, all_tensors), all_tensors);
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
