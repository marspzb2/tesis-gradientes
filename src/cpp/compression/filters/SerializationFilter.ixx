//
// Created by Marcos Perez on 6/6/26.
//

module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/tensor.h>
#include <spdlog/spdlog.h>
#include <functional>

export module SerializationFilter;
import CompressionFilter;
import torch_scalar_utils;
import TensorSerialization;
using at::Tensor;

export namespace compression::filters {
    template<typename InputT, typename OutputT, typename EncMetaT>
    class FixedSerializationForwardFilter : public base::CompressionFilter<InputT, OutputT, base::NoValue, EncMetaT> {
    private:
        std::unique_ptr<serialization::SerializationMethod<InputT, OutputT> > serialization_method;

    public:
        explicit FixedSerializationForwardFilter(
            std::unique_ptr<serialization::SerializationMethod<InputT, OutputT> > serialization_method)
            : serialization_method(std::move(serialization_method)) {
        }

        std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                             base::CompressionFilterCtx &compression_ctx) override {
            auto serialized = serialization_method->serialize(tensor, tensor.numel());
            return {
                serialized.to(tensor.device()),
                at::cat({
                    at::tensor({tensor.numel()}, meta_tensor.options()),
                    meta_tensor
                })
            };
        }
    };

    template<typename InputT, typename OutputT, typename EncMetaT>
    class FixedDeserializationBackwardFilter : public base::CompressionFilter<OutputT, InputT, EncMetaT> {
    private:
        std::unique_ptr<serialization::SerializationMethod<InputT, OutputT> > serialization_method;

    public:
        explicit FixedDeserializationBackwardFilter(
            std::unique_ptr<serialization::SerializationMethod<InputT, OutputT> > serialization_method)
            : serialization_method(std::move(serialization_method)) {
        }

        std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                             base::CompressionFilterCtx &compression_ctx) override {
            size_t size = meta_tensor[0].item<int64_t>();

            auto deserialized = serialization_method->deserialize(tensor, size);
            return {
                deserialized.to(tensor.device()),
                meta_tensor.slice(0, 1)
            };
        }
    };

    template<typename InputT, typename OutputT, typename EncMetaT>
    inline constexpr auto f_serialize_values_forward = base::FilterCtor<FixedSerializationForwardFilter<InputT, OutputT,
        EncMetaT> >{};

    template<typename InputT, typename OutputT, typename EncMetaT>
    inline constexpr auto f_serialize_values_backward = base::FilterCtor<FixedDeserializationBackwardFilter<InputT,
        OutputT, EncMetaT> >{};
}
