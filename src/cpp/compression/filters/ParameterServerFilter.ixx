module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/tensor.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/serialize/tensor.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>


export module ParameterServerFilter;
import CompressionFilter;
import Network;
import torch_scalar_utils;

using at::Tensor;
export namespace compression::filters {
    using base::CompressionFilter;


    //   Input                            InputT[] ----- reduce ---- InputT
    //     |                              |                            |
    // worker ---> MasterMessageT ---> master   < ----------------------
    //      ^                             |
    //      |---- WorkerRecvT -------------
    // ResultT
    template<
        typename InputT,
        typename FilterMetaT,
        typename SendMasterMessageT,
        typename SendMasterMetaT,
        typename SendWorkerT,
        typename SendWorkerMetaT>
    class ParameterServerFilter : public CompressionFilter<InputT, InputT, base::NoValue, FilterMetaT> {
        using WorkerEncoderFilterT = CompressionFilter<InputT, SendMasterMessageT, base::NoValue, FilterMetaT,
            SendMasterMetaT>;
        using WorkerDecoderFilterT = CompressionFilter<SendWorkerT, InputT, base::NoValue, SendWorkerMetaT, FilterMetaT>
        ;
        using MasterEncoderFilterT = CompressionFilter<InputT, SendWorkerT, base::NoValue, FilterMetaT, SendWorkerMetaT>
        ;
        bool same_size;
        std::unique_ptr<WorkerEncoderFilterT> worker_encoder;
        std::unique_ptr<base::ReductionStrategy> master_reduction_strategy;
        std::unique_ptr<MasterEncoderFilterT> master_encoder;
        std::unique_ptr<WorkerDecoderFilterT> worker_decoder;
        bool use_uncompressed_master_gradient_;
        bool use_uncompressed_reduced_master_gradient_{false};

    public:
        explicit ParameterServerFilter(
            std::unique_ptr<WorkerEncoderFilterT> worker_encoder,
            std::unique_ptr<WorkerDecoderFilterT> worker_decoder,
            std::unique_ptr<base::ReductionStrategy> master_reduction_strategy,
            std::unique_ptr<MasterEncoderFilterT> master_encoder,
            bool use_uncompressed_master_gradient = false) : worker_encoder(std::move(worker_encoder)),
                                                             worker_decoder(std::move(worker_decoder)),
                                                             master_reduction_strategy(
                                                                 std::move(master_reduction_strategy)),
                                                             master_encoder(std::move(master_encoder)),
                                                             use_uncompressed_master_gradient_(
                                                                 use_uncompressed_master_gradient) {
        }


        c10d::BroadcastOptions broadcast_options(const std::reference_wrapper<Network> &network) {
            return c10d::BroadcastOptions(network.get().master_rank());
        }

        std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                             base::CompressionFilterCtx &compression_ctx) override {
            auto network = compression_ctx.network();
            if (network.get().is_master()) {
                auto tensors = network.get().receive_in_master_with_sizes(2,
                                                                          0,
                                                                          std::vector<c10::ScalarType>{
                                                                              utils::get_torch_dtype<
                                                                                  SendMasterMessageT>(),
                                                                              utils::get_torch_dtype<SendMasterMetaT>(),
                                                                          },
                                                                          tensor.device());
                if (use_uncompressed_master_gradient_) {
                    tensors[0][network.get().master_rank()] = tensor;
                    tensors[1][network.get().master_rank()] = meta_tensor;
                } else {
                    auto [encoded_tensor,encoded_meta] = worker_encoder->
                            do_filter(tensor, meta_tensor, compression_ctx);
                    tensors[0][network.get().master_rank()] = encoded_tensor;
                    tensors[1][network.get().master_rank()] = encoded_meta;
                }

                auto [tensor,meta] = master_reduction_strategy->reduce(tensors, compression_ctx);
                auto [broadcast_tensor,broadcast_meta_tensor] = master_encoder->
                        do_filter(tensor, meta, compression_ctx);
                //Transfer lengths
                auto work = network.get().broadcast_from_master(broadcast_options(network), true,
                                                                broadcast_tensor,
                                                                broadcast_meta_tensor);
                work->wait();
                if (use_uncompressed_reduced_master_gradient_) {
                    return {tensor, meta};
                } else {
                    return worker_decoder->do_filter(broadcast_tensor, broadcast_meta_tensor, compression_ctx);
                }
            } else {
                auto [send_tensor,send_meta] = worker_encoder->do_filter(tensor, meta_tensor, compression_ctx);
                auto work = network.get().send_to_master_with_sizes(0, send_tensor, send_meta);


                work->wait();
                auto tensors = network.get().receieve_broadcast(
                    2, broadcast_options(network), std::vector{
                        utils::get_torch_dtype<SendWorkerT>(),
                        utils::get_torch_dtype<SendWorkerMetaT>()
                    }, send_tensor.device());
                return worker_decoder->do_filter(tensors[0], tensors[1], compression_ctx);
            }
        }
    };
}
