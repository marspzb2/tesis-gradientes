module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/tensor.h>
#include <ATen/TensorOperators.h>
#include <string>
#include <cmath>
#include <stdexcept>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/rand.h>
#include <torch/types.h>

export module RMQMatrixInitFilter;
import MatrixInitFilter;
import CompressionFilter;
export namespace compression::filters {
    using base::SavedMatrixState;
    using at::Tensor;

    class RMQMAtrixSampler : public MatrixSampler {
        int64_t dim_;

    public:
        RMQMAtrixSampler(int64_t dim) : dim_(dim) {
        }

        Tensor sample_matrix(const std::vector<int64_t> &shape,
                             const std::optional<at::Generator> &shared_randomness,
                             const c10::TensorOptions &options) override {
            auto generator = shared_randomness.value();
            int64_t d = this->dim_;

            if (d < 2 || (d & (d - 1)) != 0)
                throw std::invalid_argument("d < 2");
            double k = std::log2(d);

            Tensor H = walsh(k, options.dtype(torch::kFloat32).device(torch::kCPU));
            const Tensor D = at::mul(
                (at::rand(H.sizes(), generator, H.options()) > 0.5) * 2 - 1,
                at::eye(d, H.options()));

            auto sampled_matrix = (1 / std::sqrt(d * 1.)) * at::matmul(H, D);
            return sampled_matrix.to(options.device());
        }

        Tensor walsh(int k, const at::TensorOptions &options) {
            auto h = haddamard(k, options);
            auto left_side = h.slice(1, 0, -1);
            auto right_side = h.slice(1, 1, h.size(1));
            auto sign_changes = (left_side * right_side == -1).sum(1);
            auto sorted_row_indices = at::argsort(sign_changes);
            return h.index_select(0, sorted_row_indices);
        }

        Tensor haddamard(int k, const at::TensorOptions &options) {
            if (k == 1) {
                return at::tensor({1, 1, 1, -1}, options).reshape({2, 2});
            } else {
                return at::kron(haddamard(1, options), haddamard(k - 1, options));
            }
        }
    };

    template<typename ValueT, typename EncT>
    auto f_rmq_matrix_init(
        std::string matrix_name,
        int dim) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncT> >(
            std::move(matrix_name),
            std::vector<int64_t>{},
            std::make_unique<RMQMAtrixSampler>(dim),
            true,
            false,
            true);
    }

    template<typename ValueT, typename EncT>
    auto f_rmq_matrix_recover(
        std::string matrix_name,
        int dim) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncT> >(
            std::move(matrix_name),
            std::vector<int64_t>{},
            std::make_unique<RMQMAtrixSampler>(dim),
            true,
            true,
            true);
    }

    template<typename ValueT, typename EncT>
    auto f_rmq_rotate_operation_forward(std::string rotate_matrix_name,
                            std::optional<std::string> input_matrix = std::nullopt,
                            bool pass_result = false,
                            std::optional<std::string> save_matrix_name = std::nullopt) {
        return base::MapFilter<ValueT, EncT>(
            [rotate_matrix_name, input_matrix](const Tensor &tensor, base::CompressionFilterCtx &compression_ctx) {
                Tensor input_tensor = tensor;
                if (input_matrix.has_value()) {
                    input_tensor = compression_ctx.get_state<SavedMatrixState>(input_matrix.value())->matrix();
                }
                auto rotation_matrix_state = compression_ctx.get_state<SavedMatrixState>(rotate_matrix_name);
                auto rotation_matrix = rotation_matrix_state->matrix();
                return input_tensor.reshape({-1, rotation_matrix.size(0)})
                        .matmul(rotation_matrix.to(input_tensor.device()).t())
                        .flatten();
            },
            pass_result,
            save_matrix_name
        );
    }


    template<typename ValueT, typename EncT>
    auto f_rmq_rotate_operation_backward(std::string rotate_matrix_name,
                                bool pass_result = false,
                                std::optional<std::string> save_matrix_name = std::nullopt) {
        return base::MapFilter<ValueT, EncT>(
            [rotate_matrix_name](const Tensor &input_tensor, base::CompressionFilterCtx &compression_ctx) {
                auto rotation_matrix_state = compression_ctx.get_state<SavedMatrixState>(rotate_matrix_name);
                auto rotation_matrix = rotation_matrix_state->matrix();
                return input_tensor.reshape({-1, rotation_matrix.size(0)})
                        .matmul(rotation_matrix.to(input_tensor.device()))
                        .flatten();
            },
            pass_result,
            save_matrix_name
        );
    }
}
