//
// Created by Marcos Perez on 7/6/26.
//
module;
#include  <torch/torch.h>

export module MatrixInitFilter;
import CompressionFilter;
using torch::Tensor;
using namespace compression::base;

export
{
    namespace compression::filters {
        enum class MatrixInitializationDistribution {
            UNIFORM,
            GAUSS,
            FIXED
        };

        template<typename ValueT, typename EncMetaT>
        class MatrixInitializationFilter : public CompressionFilter<ValueT, ValueT, NoValue, EncMetaT, EncMetaT,
                    SavedMatrixState> {
        private:
            std::string matrix_name;
            MatrixInitializationDistribution init_distribution = MatrixInitializationDistribution::UNIFORM;
            std::vector<int64_t> shape;
            std::optional<Tensor> base_matrix{};
            bool shared_seed = false;
            bool sample_shared_seed = false;
            double mean = 0.0;
            double std = 1.0;
            bool reinitialize = false;

        public:
            MatrixInitializationFilter(const std::string &matrix_name,
                                       const std::optional<Tensor> &base_matrix,
                                       std::vector<int64_t> shape,
                                       MatrixInitializationDistribution init_distribution,
                                       bool shared_seed = false,
                                       bool sample_shared_seed = false,
                                       double mean = 0.0,
                                       double std = 1.0,
                                       bool reinitialize = false
            )
                : matrix_name(matrix_name),
                  shape(shape),
                  base_matrix(base_matrix),
                  init_distribution(init_distribution),
                  shared_seed(shared_seed),
                  sample_shared_seed(sample_shared_seed),
                  mean(mean),
                  std(std),
                  reinitialize(reinitialize) {
            }

            static auto initialize_fixed_matrix(
                const std::string &matrix_name,
                const Tensor &base_matrix) {
                return std::make_unique<MatrixInitializationFilter<ValueT, EncMetaT> >(
                    matrix_name, base_matrix,
                    std::vector<int64_t>(base_matrix.sizes().begin(), base_matrix.sizes().end()),
                    MatrixInitializationDistribution::FIXED);
            }


            static auto initialize_uniform_matrix(
                const std::string &matrix_name,
                const std::vector<int64_t> &shape,
                bool reinitialize = false,
                bool shared_seed = false,
                bool sample_shared_seed = false
            ) {
                return std::make_unique<MatrixInitializationFilter<ValueT, EncMetaT> >(matrix_name, std::nullopt,
                    std::vector<int64_t>{shape}, MatrixInitializationDistribution::UNIFORM, shared_seed,
                    sample_shared_seed, 0, 0, reinitialize);
            }

            static auto initialize_gauss_matrix(
                const std::string &matrix_name,
                const std::vector<int64_t> &shape,
                double mean,
                double std,
                bool shared_seed = false,
                bool sample_shared_seed = false,
                bool reinitialize = false
            ) {
                return std::make_unique<MatrixInitializationFilter<ValueT, EncMetaT> >(matrix_name, std::nullopt,
                    std::vector<int64_t>{shape}, MatrixInitializationDistribution::UNIFORM, shared_seed,
                    sample_shared_seed, mean, std, reinitialize);
            }


            std::unique_ptr<SavedMatrixState>
            create_state(const Tensor &tensor, const Tensor &previousMetadata) override {
                if (base_matrix.has_value()) {
                    return std::make_unique<SavedMatrixState>(matrix_name, base_matrix.value());
                }
                std::vector<int64_t> real_shape{};
                for (int64_t size: this->shape) {
                    if (size < 0) {
                        real_shape.push_back(tensor.sizes()[size * -1 - 1]);
                    } else {
                        real_shape.push_back(size);
                    }
                }
                if (this->init_distribution == MatrixInitializationDistribution::UNIFORM) {
                    return std::make_unique<SavedMatrixState>(matrix_name, torch::rand(real_shape, tensor.options()));
                }
                if (this->init_distribution == MatrixInitializationDistribution::GAUSS) {
                    return std::make_unique<SavedMatrixState>(matrix_name,
                                                              torch::normal(
                                                                  mean, std, real_shape, std::nullopt,
                                                                  tensor.options()));
                }
                throw std::logic_error("SavedMatrixState::create(): not implemented");
            }


            bool recreate_state(const Tensor &tensor, const Tensor &meta_tensor,
                                const CompressionFilterCtx &compression_ctx) override {
                return this->reinitialize;
            }

            std::string state_name(const Tensor &tensor) override {
                return matrix_name;
            }

            bool has_state(const Tensor &tensor, const Tensor &previousMetadata,
                           const std::optional<NoValue> &metadata) override {
                return true;
            }
        };
    }
}
