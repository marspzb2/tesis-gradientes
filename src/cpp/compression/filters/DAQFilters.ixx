module;
#include <memory>
#include <ATen/core/Tensor.h>
#include <ATen/ops/tensor.h>
#include <ATen/TensorOperators.h>
#include <string>
#include <cmath>
#include <stdexcept>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/rand.h>
#include <torch/types.h>
#include <vector>

export module DAQFilters;
import MatrixInitFilter;
import CompressionFilter;
import torch_scalar_utils;
export namespace compression::filters {
    using base::SavedMatrixState;
    using at::Tensor;

    class DAQMAtrixSampler : public MatrixSampler {
        int64_t dim_;
        int64_t max_k_;
        Tensor scales_tensor;

    public:
        DAQMAtrixSampler(int64_t dim, const std::vector<double> &scales) : dim_(dim),
                                                                           scales_tensor(at::tensor(scales)),
                                                                           max_k_(scales.size()) {
        }

        Tensor sample_matrix(const std::vector<int64_t> &shape,
                             const std::optional<at::Generator> &shared_randomness,
                             const c10::TensorOptions &options) override {
            auto generator = shared_randomness.value();
            int64_t d = this->dim_;

            std::vector<int64_t> final_shape;
            if (d < 0) {
                final_shape = std::vector<int64_t>(shape);
                final_shape.push_back(max_k_);
            } else {
                final_shape = std::vector<int64_t>{shape[0] / dim_, dim_, max_k_};
            }
            auto sampled = at::rand(final_shape, generator, options.dtype(torch::kFloat)) * 2 - 1; // sample unitary.


            sampled = sampled * scales_tensor;
            return sampled;
        }
    };

    template<typename ValueT, typename EncT>
    auto f_daq_matrix_init(std::string matrix_name,
                       int dim,
                       const std::vector<double> &scales = {1}) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncT> >(
            std::move(matrix_name),
            std::vector<int64_t>{-1, (int) scales.size()},
            std::make_unique<DAQMAtrixSampler>(dim, scales),
            true,
            false,
            true);
    }

    template<typename ValueT, typename EncT>
    auto f_daq_matrix_recover(
        std::string matrix_name,
        int dim,
        const std::vector<double> &scales) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncT> >(
            std::move(matrix_name),
            std::vector<int64_t>{},
            std::make_unique<DAQMAtrixSampler>(dim, scales),
            true,
            true,
            true);
    }

    template<typename ValueT, typename EncodedT, typename MetaEncT>
    auto f_daq_encode_forward(std::string matrix_name, int64_t max_k = 1) {
        return base::MapFilter<ValueT, MetaEncT, EncodedT>(
            [matrix_name,max_k](const Tensor &input_tensor, base::CompressionFilterCtx &compression_ctx) {
                auto sampled_matrix_state = compression_ctx.get_state<SavedMatrixState>(matrix_name);
                auto sampled_matrix = sampled_matrix_state->matrix();
                return ((input_tensor.reshape({-1, 1}) > sampled_matrix.reshape({-1, max_k})))
                        .to(utils::get_torch_dtype<EncodedT>())
                        .flatten();
            },
            true,
            std::nullopt
        );
    }

    template<typename ValueT, typename EncodedT, typename MetaEncT>
    auto f_daq_decode(std::string sample_matrix_name,
                   std::string reference_matrix_name,
                   int64_t max_k = 1) {
        return base::MapFilter<EncodedT, MetaEncT, ValueT>(
            [sample_matrix_name,max_k,reference_matrix_name](const Tensor &input_tensor,
                                                             base::CompressionFilterCtx &compression_ctx) {
                auto device = input_tensor.device();
                auto sampled_matrix_state = compression_ctx.get_state<SavedMatrixState>(sample_matrix_name);
                auto sampled_matrix = sampled_matrix_state->matrix()
                        .to(device);
                auto reference_matrix = compression_ctx.get_state<SavedMatrixState>(reference_matrix_name)->matrix()
                        .to(device)
                        .reshape({-1, 1});
                auto reference_matrix_reconstruction =
                        (reference_matrix > sampled_matrix.reshape({-1, max_k}))
                        .to(utils::get_torch_dtype<ValueT>());

                return ((2 * (input_tensor.reshape({-1, 1}) - reference_matrix_reconstruction)) + reference_matrix)
                        .to(utils::get_torch_dtype<ValueT>())
                        .flatten();
            },
            true,
            std::nullopt
        );
    }
}
