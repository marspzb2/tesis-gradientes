//
// Created by Marcos Perez on 7/6/26.
//
module;

#include <ATen/ops/tensor.h>
#include <ATen/TensorOperators.h>
#include <string>
#include <cmath>
#include <stdexcept>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/ops/rand.h>
#include <torch/types.h>

export module MatrixInitFilter;
import CompressionFilter;
using at::Tensor;
using at::Generator;
using namespace compression::base;

export namespace compression::filters {
    class MatrixSampler {
    public:
        virtual ~MatrixSampler() = default;

        virtual Tensor sample_matrix(const std::vector<int64_t> &shape,
                                     const std::optional<Generator> &shared_randomness,
                                     const c10::TensorOptions &options) =0;
    };

    class UniformSampler : public MatrixSampler {
    private:
        double min_;
        double max_;

    public:
        UniformSampler(double min, double max)
            : min_(min),
              max_(max) {
        }

        Tensor sample_matrix(const std::vector<int64_t> &shape,
                             const std::optional<Generator> &generator,
                             const c10::TensorOptions &options) override {
            return (at::rand(shape, generator, options) * 2 * max_) / (max_ - min_);
        }
    };

    class GaussianSampler : public MatrixSampler {
    private:
        double mean_;
        double std_;

    public:
        GaussianSampler(double mean, double std) : mean_(mean), std_(std) {
        }

        Tensor sample_matrix(const std::vector<int64_t> &shape, const std::optional<Generator> &generator,
                             const c10::TensorOptions &options) override {
            return at::rand(shape, generator, options);
        }
    };

    template<typename ValueT, typename EncMetaT>
    class MatrixInitializationFilter : public CompressionFilter<ValueT, ValueT, at::Generator, EncMetaT> {
    protected:
        std::string matrix_name_;
        std::vector<int64_t> shape_;
        std::unique_ptr<MatrixSampler> sampler_;
        bool share_randomness;
        bool recover_;
        bool reinitialize = false;
        int initial_seed;

    public:
        ~MatrixInitializationFilter() override = default;

        MatrixInitializationFilter(const std::string &matrix_name,
                                   const std::vector<int64_t> &shape,
                                   std::unique_ptr<MatrixSampler> sampler,
                                   bool share_randomness,
                                   bool recover,
                                   bool reinitialize)
            : matrix_name_(matrix_name),
              shape_(shape),
              sampler_(std::move(sampler)),
              share_randomness(share_randomness),
              reinitialize(reinitialize),
              recover_(recover) {
        }


        virtual bool can_create_state(const Tensor &tensor, const std::optional<Generator> &metadata,
                                      CompressionFilterCtx &ctx) {
            return true;
        }

        virtual Tensor apply_matrix(const Tensor &og_tensor,
                                    const Tensor &sampled_matrix,
                                    const std::optional<Generator> &metadata,
                                    CompressionFilterCtx &ctx
        ) {
            return og_tensor;
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<Generator> &metadata,
                                  NoValue *state, CompressionFilterCtx &ctx) override {
            if (can_create_state(tensor, metadata, ctx) && (!ctx.is_state_present(matrix_name_) || reinitialize)) {
                std::vector<int64_t> real_shape;
                if (this->shape_.empty()) {
                    real_shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
                } else {
                    for (int64_t size: this->shape_) {
                        if (size < 0) {
                            real_shape.push_back(tensor.sizes()[size * -1 - 1]);
                        } else {
                            real_shape.push_back(size);
                        }
                    }
                }
                auto options = tensor.options();
                auto new_matrix = sampler_->sample_matrix(real_shape, metadata, options);
                ctx.put_state(matrix_name_, std::make_unique<SavedMatrixState>(matrix_name_, new_matrix));
            }
            return apply_matrix(tensor, ctx.get_state<SavedMatrixState>(matrix_name_)->matrix(), metadata, ctx);
        }

        Tensor internal_metadata(const Tensor &tensor, const Tensor &meta_tensor,
                                 const std::optional<Generator> &metadata, NoValue *state,
                                 CompressionFilterCtx &ctx) override {
            if (share_randomness && recover_) {
                return meta_tensor.slice(0, 1);
            } else if (share_randomness) {
                return at::cat({
                    at::tensor({initial_seed}, meta_tensor.options()),
                    meta_tensor
                });
            } else {
                return meta_tensor;
            }
        }

        std::optional<Generator> parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) override {
            if (share_randomness && recover_) {
                return std::optional(at::detail::createCPUGenerator(previousMetadata[0].item<int64_t>()));
            } else if (share_randomness) {
                this->initial_seed = torch::randint(10000, {1}).item<int64_t>();
                return std::optional(at::detail::createCPUGenerator(initial_seed));
            } else {
                return std::nullopt;
            }
        }
    };

    template<typename ValueT, typename EncMetaT>
    auto UniformMatrixInitialization(
        std::string matrix_name,
        std::vector<int64_t> shape,
        double min = 0,
        double max = 1,
        bool share_randomness = false,
        bool recover = false,
        bool reinitialize = false) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncMetaT> >(
            std::move(matrix_name), shape,
            std::make_unique<UniformSampler>(min, max),
            share_randomness, recover, recover || reinitialize
        );
    }

    template<typename ValueT, typename EncMetaT>
    auto GaussianMatrixInitialization(
        std::string matrix_name,
        std::vector<int64_t> shape,
        double mean = 0,
        double std = 1,
        bool share_randomness = false,
        bool recover = false,
        bool reinitialize = false) {
        return std::make_unique<MatrixInitializationFilter<ValueT, EncMetaT> >(
            std::move(matrix_name), shape,
            std::make_unique<GaussianSampler>(mean, std),
            share_randomness, recover, recover || reinitialize
        );
    }
}
