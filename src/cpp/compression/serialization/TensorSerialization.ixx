module;
#include <algorithm>
#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <concepts>
#include <cassert>
#include <limits>
#include <memory>
#include <type_traits>
#include <bit>
#include <vector>
#include <omp.h>
export module TensorSerialization;
import torch_scalar_utils;
import BitUtils;
using torch::Tensor;

export namespace compression::serialization {
    template<typename InputT, typename OutputT>
    class SerializationMethod {
    public:
        virtual ~SerializationMethod()  =default;

        virtual Tensor serialize(const Tensor &input, size_t qty) {
            Tensor result = torch::zeros({initial_size_estimation(input, qty)},
                                         torch::TensorOptions()
                                         .device(torch::kCPU)
                                         .dtype(utils::get_torch_dtype<OutputT>()));
            size_t serialized_qty = this->serialize(input.to(torch::kCPU).contiguous().flatten().data_ptr<InputT>(),
                                                 result.data_ptr<OutputT>(),
                                                 qty);
            return result.slice(0, 0, serialized_qty).to(input.device());
        }

        virtual Tensor deserialize(const Tensor &input, int size) {
            Tensor result = torch::zeros({size},
                                         torch::TensorOptions()
                                         .device(torch::kCPU)
                                         .dtype(utils::get_torch_dtype<InputT>()));
            this->deserialize(input.to(torch::kCPU).contiguous().data_ptr<OutputT>(), result.data_ptr<InputT>(), size);

            return result.to(input.device());
        }

        virtual int64_t initial_size_estimation(const Tensor &input, int64_t qty) {
            return qty;
        }

        virtual size_t serialize(InputT *source_pointer, OutputT *result_pointer, size_t quantity) =0;

        virtual size_t deserialize(OutputT *serialized_pointer, InputT *deserialized_pointer, size_t quantity) =0;

        virtual std::unique_ptr<SerializationMethod<InputT, OutputT>> clone()=0;

    };

}
