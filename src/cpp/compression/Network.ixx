//
// Created by Marcos Perez on 4/6/26.
//
module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/TensorOptions.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/serialize/tensor.h>
#include <atomic>
#include <concepts>

export module Network;
import CommonConcepts;
import CommonValidations;

export enum class Topology {
    STAR,
    RING
};

using at::Tensor;
using compression::utils::validate_positive;
using compression::utils::validate_same_size;
using compression::utils::validate_size_is;
using compression::utils::validate_range;
using compression::utils::validate_flat_tensor;
using compression::utils::validate_positive_or_zero;
using compression::utils::validate_gte;
using compression::utils::TensorT;


export class WorkCombination {
private:
    std::vector<c10::intrusive_ptr<c10d::Work> > works;

public:
    explicit WorkCombination(const std::vector<c10::intrusive_ptr<c10d::Work> > &works)
        : works(works) {
    }

    void wait() {
        for (const auto &work: works) {
            work->wait();
        }
    }
};


export class Network {
private:
    c10::intrusive_ptr<c10d::ProcessGroup> process_group;
    std::optional<Topology> topology;
    std::atomic<uint64_t> send_bytes{0};
    std::atomic<uint64_t> recv_bytes{0};

    Network() = default;

public:
    Network(const Network &) = delete;

    Network &operator=(const Network &) = delete;

    static Network &instance() {
        static Network network;
        return network;
    }

    [[nodiscard]] uint64_t get_sent_bytes() const {
        return this->send_bytes.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t get_recv_bytes() const {
        return this->recv_bytes.load(std::memory_order_relaxed);
    }

    [[nodiscard]] int current_rank() const {
        return this->process_group->getRank();
    }

    void reset_network_stats() {
        this->send_bytes.store(0, std::memory_order_relaxed);
        this->recv_bytes.store(0, std::memory_order_relaxed);
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

    c10::intrusive_ptr<c10d::Work> _all_gather(std::vector<Tensor> &inputs,
                                               std::vector<std::vector<Tensor> > &outputs) {
        return this->process_group->allgather(outputs, inputs, c10d::AllgatherOptions());
    }

    std::vector<std::vector<Tensor> > make_send_copies_same_size(const std::vector<Tensor> &send_tensors,
                                                                 int world_size) {
        std::vector<std::vector<Tensor> > all_tensors(send_tensors.size());
        int send_idx = 0;
        for (const Tensor &send_tensor: send_tensors) {
            all_tensors[send_idx] = std::vector<Tensor>();
            // transfer_list[send_idx] = send_tensor;
            for (int j = 0; j < world_size; j++) {
                all_tensors[send_idx].push_back(at::empty_like(send_tensor));
            }
            send_idx++;
        }
        return all_tensors;
    }


    template<typename... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<Tensor> > > all_gather_same_size(
        Args &&... tensors) {
        int world_size = process_group->getSize();
        std::vector<Tensor> send_tensors{std::forward<Args>(tensors)...};
        std::vector<std::vector<Tensor> > all_tensors = make_send_copies_same_size(send_tensors, world_size);
        this->add_recv_bytes(all_tensors, current_rank());
        this->add_send_bytes(send_tensors, world_size - 1);
        return std::make_tuple(this->_all_gather(send_tensors, all_tensors), all_tensors);
    }

    template<TensorT... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<Tensor> > > all_gather(
        bool same_size = true, Args &&... args) {
        if (same_size) {
            return all_gather_same_size(std::forward<Args>(args)...);
        } else {
            std::vector<Tensor> send_tensors{std::forward<Args>(args)...};
            validate_size_is(send_tensors, 1, "Variable-size all_gather supports one tensor per call");

            Tensor send_tensor = send_tensors.front();
            validate_flat_tensor(send_tensor, "Variable-size all_gather expects a 1D tensor");

            int world_size = process_group->getSize();

            auto [size_work, gathered_sizes] =
                    this->all_gather_sizes(send_tensor);
            size_work->wait();

            std::vector<int64_t> sizes;
            sizes.reserve(world_size);
            int64_t max_size = 0;
            for (const Tensor &size_tensor: gathered_sizes.front()) {
                int64_t size = size_tensor[0].item<int64_t>();
                sizes.push_back(size);
                max_size = std::max(max_size, size);
            }

            Tensor padded = at::zeros({max_size}, send_tensor.options());
            padded.slice(0, 0, send_tensor.numel()).copy_(send_tensor);

            std::vector<Tensor> padded_inputs{padded};
            auto gathered_padded =
                    make_send_copies_same_size(padded_inputs, world_size);
            auto work = this->_all_gather(padded_inputs, gathered_padded);
            work->wait();

            // Count the padded buffers that actually crossed the network.
            this->add_recv_bytes(gathered_padded, current_rank());
            this->add_send_bytes(padded_inputs, world_size - 1);

            for (int rank = 0; rank < world_size; ++rank) {
                gathered_padded[0][rank] =
                        gathered_padded[0][rank].slice(0, 0, sizes[rank]);
            }

            return std::make_tuple(work, std::move(gathered_padded));
        }
    }

    template<TensorT... Args>
    std::tuple<c10::intrusive_ptr<c10d::Work>, std::vector<std::vector<Tensor> > > all_gather_sizes(
        Args &&... tensors) {
        std::vector<Tensor> sizes_tensors{
            at::tensor(tensors.sizes(), at::TensorOptions().device(tensors.device()).dtype(at::kLong))...
        };
        int world_size = process_group->getSize();
        std::vector<std::vector<Tensor> > all_tensors = make_send_copies_same_size(sizes_tensors, world_size);
        add_recv_bytes(all_tensors, current_rank());
        add_send_bytes(sizes_tensors, world_size - 1);
        return std::make_tuple(this->_all_gather(sizes_tensors, all_tensors), all_tensors);
    }

    template<TensorT... Args>
    c10::intrusive_ptr<c10d::Work> all_reduce(Args &&... tensors) {
        return this->all_reduce_with_op(
            c10d::ReduceOp::SUM,
            std::forward<Args>(tensors)...
        );
    }

    template<TensorT... Args>
    c10::intrusive_ptr<c10d::Work> all_reduce_with_op(
        c10d::ReduceOp::RedOpType operation,
        Args &&... tensors) {
        std::vector<Tensor> send_tensors{std::forward<Args>(tensors)...};

        c10d::AllreduceOptions options;
        options.reduceOp = operation;
        add_recv_bytes_fraction(send_tensors, 2 * (world_size() - 1), world_size());
        add_send_bytes_fraction(send_tensors, 2 * (world_size() - 1), world_size());
        return this->process_group->allreduce(send_tensors, options);
    }

    void add_send_bytes(const std::vector<std::vector<Tensor> > &tensors_by_value, int ignore_rank) {
        for (const auto &tensors_by_rank: tensors_by_value) {
            for (int rank = 0; rank < tensors_by_rank.size(); ++rank) {
                if (rank != ignore_rank) {
                    this->add_send_bytes(std::vector<Tensor>{tensors_by_rank[rank]});
                }
            }
        }
    }

    void add_send_bytes(const std::vector<Tensor> &send_tensors, int multiply = 1) {
        uint64_t total_bytes = 0;
        for (const auto &tensor: send_tensors) {
            total_bytes += tensor.numel() * tensor.element_size() * multiply;
        }
        this->send_bytes.fetch_add(total_bytes, std::memory_order_relaxed);
    }

    void add_recv_bytes(const std::vector<std::vector<Tensor> > &tensors_by_value, int ignore_rank) {
        for (const auto &tensors: tensors_by_value) {
            for (int rank = 0; rank < tensors.size(); ++rank) {
                if (rank != ignore_rank) {
                    this->add_recv_bytes(std::vector{tensors[rank]});
                }
            }
        }
    }

    void add_recv_bytes(const std::vector<Tensor> &send_tensors, int amount = 1) {
        uint64_t total_bytes = 0;
        for (const auto &tensor: send_tensors) {
            total_bytes += tensor.numel() * tensor.element_size() * amount;
        }
        this->recv_bytes.fetch_add(total_bytes, std::memory_order_relaxed);
    }

    uint64_t bytes_in(const std::vector<Tensor> &tensors) const {
        uint64_t total_bytes = 0;
        for (const auto &tensor: tensors) {
            total_bytes += static_cast<uint64_t>(tensor.numel())
                    * static_cast<uint64_t>(tensor.element_size());
        }
        return total_bytes;
    }

    void add_send_bytes_fraction(const std::vector<Tensor> &tensors,
                                 uint64_t numerator,
                                 uint64_t denominator) {
        const uint64_t added_bytes = bytes_in(tensors) * numerator / denominator;
        this->send_bytes.fetch_add(added_bytes, std::memory_order_relaxed);
    }

    void add_recv_bytes_fraction(const std::vector<Tensor> &tensors,
                                 uint64_t numerator,
                                 uint64_t denominator) {
        const uint64_t added_bytes = bytes_in(tensors) * numerator / denominator;
        this->recv_bytes.fetch_add(added_bytes, std::memory_order_relaxed);
    }


    template<TensorT... Args>
    Tensor to_send_sizes(Args &&... tensors) {
        static_assert(sizeof...(Args) > 0);

        std::vector<Tensor> send_tensors{
            std::forward<Args>(tensors)...
        };

        std::vector<int64_t> sizes;
        for (const auto &tensor: send_tensors) {
            sizes.push_back(tensor.numel());
        }
        Tensor send_sizes = torch::tensor(sizes,
                                          at::TensorOptions()
                                          .dtype(torch::kLong)
                                          .device(send_tensors.front().device()));
        return send_sizes;
    }

    template<TensorT... Args>
    auto send_to_master(int tag, Args &&... tensors) {
        std::vector<c10::intrusive_ptr<c10d::Work> > works;
        //Por compatibilidad hay que mandar uno a uno
        auto tensor_tuple = std::forward_as_tuple(tensors...);
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            ([&]() {
                auto index = Is;
                Tensor value_to_send = std::get<Is>(tensor_tuple);
                std::vector<Tensor> send_tensors{value_to_send};
                add_send_bytes(send_tensors);
                works.push_back(
                    this->process_group->send(send_tensors,
                                              master_rank(),
                                              tag + index
                    )
                );
            }(), ...);
        }(std::make_index_sequence<sizeof...(Args)>{});

        return std::make_unique<WorkCombination>(works);
    }

    template<TensorT... Args>
    auto send_to_master_with_sizes(int base_tag, Args &&... tensors) {
        Tensor send_sizes = to_send_sizes(std::forward<Args>(tensors)...);

        auto sizes_work = send_to_master(base_tag, send_sizes);
        sizes_work->wait();

        return send_to_master(base_tag + 1, std::forward<Args>(tensors)...);
    }

    static void wait_all(const std::vector<c10::intrusive_ptr<c10d::Work> > &remaining_works) {
        for (const auto &work: remaining_works) {
            work->wait();
        }
    }

    auto receive_in_master_with_sizes(int vector_count, int base_tag,
                                      std::vector<c10::ScalarType> dtypes,
                                      at::Device device) {
        validate_positive(vector_count,"El vector count tiene que ser mayor que 0");
        validate_size_is(dtypes,vector_count,"Dtypes is wrong");
        std::vector<std::vector<Tensor> > in_sizes(1);
        in_sizes[0].resize(world_size());
        std::vector<c10::intrusive_ptr<c10d::Work> > sizes_works;
        auto &recv_line = in_sizes[0];
        for (int rank = 0; rank < world_size(); rank++) {
            Tensor node_sizes = torch::zeros({vector_count}, at::TensorOptions().dtype(torch::kLong).device(device));
            recv_line[rank] = node_sizes;
            if (rank != master_rank()) {
                auto recv_vector = std::vector{node_sizes};
                auto sizes_work = this->process_group->recv(recv_vector, rank, base_tag);
                sizes_works.push_back(sizes_work);
            }
        }
        wait_all(sizes_works);

        add_recv_bytes(in_sizes, master_rank());
        std::vector<std::vector<Tensor> > in_tensors(vector_count);

        std::vector<c10::intrusive_ptr<c10d::Work> > tensor_works;
        for (int rank = 0; rank < world_size(); rank++) {
            Tensor sizes = in_sizes[0][rank].cpu();
            for (std::size_t i = 0; i < sizes.numel(); i++) {
                auto tensor_size = sizes[i].item<int64_t>();
                validate_positive_or_zero(tensor_size,"Size is wrongly send");
                auto recv_tensor = torch::zeros({tensor_size},
                                                at::TensorOptions()
                                                .dtype(dtypes[i])
                                                .device(device));
                in_tensors[i].push_back(recv_tensor);
                if (rank != master_rank()) {
                    std::vector recv_line{recv_tensor};
                    auto data_work = this->process_group->recv(recv_line, rank, base_tag + 1 + i);
                    tensor_works.push_back(data_work);
                }
            }
        }

        wait_all(tensor_works);
        add_recv_bytes(in_tensors, master_rank());

        return in_tensors;
    }

    //template<typename... Args>
    //auto receive_in_master(c10d::GatherOptions opts, Args &&... tensors) {
    //    std::vector<Tensor> send_tensors{std::forward<Args>(tensors)...};
    //    std::vector<std::vector<Tensor> > recv_tensors{};
    //    for (const auto &tensor: send_tensors) {
    //        std::vector<Tensor> recv_tensors_from_worker;
    //        for (int i = 0; i < this->process_group->getSize(); i++) {
    //            recv_tensors_from_worker.push_back(torch::empty_like(tensor));
    //        }
    //        recv_tensors.push_back(recv_tensors_from_worker);
    //    }
    //
    //    add_recv_bytes(recv_tensors, current_rank());
    //    return std::make_tuple(
    //        this->process_group->gather(
    //            recv_tensors,
    //            send_tensors,
    //            opts),
    //        recv_tensors);
    //}

    auto receieve_broadcast(const std::vector<int64_t> &sizes_vector, const c10d::BroadcastOptions &opts,
                            std::vector<c10::ScalarType> dtypes, at::Device device) {
        std::vector<Tensor> recv_tensors;
        std::vector<c10::intrusive_ptr<c10d::Work> > tensor_works;
        validate_same_size(sizes_vector,dtypes,"Sizes is wrong wrto dtypes");
        int i = 0;
        for (const auto &size: sizes_vector) {
            auto recv_tensor = torch::zeros({size}, at::TensorOptions().dtype(dtypes[i]).device(device));
            recv_tensors.push_back(recv_tensor);
            std::vector recv_line{recv_tensor};
            tensor_works.push_back(this->process_group->broadcast(recv_line, opts));
            i++;
        }
        wait_all(tensor_works);
        add_recv_bytes(recv_tensors);
        return recv_tensors;
    }

    auto receieve_broadcast(uint8_t vector_quantity,
                            const c10d::BroadcastOptions &opts,
                            std::vector<c10::ScalarType> dtypes,
                            at::Device device) {
        Tensor sizes_vector = torch::zeros({vector_quantity},
                                           at::TensorOptions()
                                           .dtype(torch::kLong)
                                           .device(device));
        std::vector size_tensor{sizes_vector};
        auto sizes_work = this->process_group->broadcast(size_tensor, opts);
        add_recv_bytes(size_tensor);
        sizes_work->wait();
        sizes_vector = sizes_vector.cpu();
        std::vector sizes(sizes_vector.data_ptr<int64_t>(), sizes_vector.data_ptr<int64_t>() + sizes_vector.numel());
        return receieve_broadcast(sizes, opts, dtypes, device);
    }

    template<TensorT... Args>
    auto broadcast_from_master(c10d::BroadcastOptions opts, bool send_sizes, Args &&... tensors) {
        static_assert(sizeof...(Args) > 0);

        std::vector<Tensor> send_tensors{std::forward<Args>(tensors)...};
        if (send_sizes) {
            constexpr size_t broadcasted_tensor_count = sizeof...(tensors);
            Tensor sizes_tensor = torch::zeros({broadcasted_tensor_count},
                                               c10::TensorOptions()
                                               .dtype(torch::kLong)
                                               .device(send_tensors.front().device()));
            int size_idx = 0;
            for (const auto &tensor: send_tensors) {
                sizes_tensor[size_idx++] = tensor.numel();
            }
            add_send_bytes(std::vector<Tensor>{sizes_tensor}, world_size() - 1);
            std::vector send_sizes_vec{sizes_tensor};
            auto work = this->process_group->broadcast(send_sizes_vec, opts);
            work->wait();
        }
        add_send_bytes(send_tensors, world_size() - 1);
        std::vector<c10::intrusive_ptr<c10d::Work> > send_works;

        for (const Tensor &tensor: send_tensors) {
            std::vector send_line{tensor};
            send_works.push_back(
                process_group->broadcast(send_line, opts)
            );
        }

        return std::make_unique<WorkCombination>(send_works);
    }

    bool is_master() {
        return this->process_group->getRank() == master_rank();
    }

    int master_rank() {
        return 0;
    }

    int world_size() {
        return this->process_group->getSize();
    }
};
