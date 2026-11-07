module;
#include <concepts>
#include <ATen/core/Tensor.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/round.h>
#include <cstdint>
#include <ATen/ops/div.h>
#include <ATen/TensorOperators.h>
#include <concepts>
#include <stdexcept>
#include <string>
#include <utility>

export module ModuloQuantizationFilter;
import CompressionFilter;
import torch_scalar_utils;
import MatrixMultiplyFilter;
using at::Tensor;
export namespace compression::filters {
    template<typename OriginalValueT, std::integral QuantizationT>
    class ForwardModuloQuantizationFilter
            : public base::CompressionFilter<OriginalValueT, QuantizationT> {
    private:
        int64_t divisor_;

    public:
        explicit ForwardModuloQuantizationFilter(int64_t divisor)
            : divisor_(divisor) {
            if (divisor <= 0) {
                throw std::invalid_argument("divisor  must be greater than 0");
            }
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
           return at::remainder(tensor, divisor_).to(utils::get_torch_dtype<QuantizationT>());
        }
    };


    template<typename OriginalValueT, typename QuantizationT>
    class BackwardModuloQuantizationFilter
            : public base::CompressionFilter<QuantizationT, OriginalValueT> {
        int64_t divisor_;

        double lattice_size_;
        std::string proximity_name_;
        MatrixTransformType transform_type_;

    public:
        explicit BackwardModuloQuantizationFilter(int64_t divisor,
                                                  double lattice_size,
                                                  std::string proximity_name)
            : divisor_(divisor), lattice_size_(lattice_size), proximity_name_(std::move(proximity_name)),
              transform_type_(MatrixTransformType::NONE) {
            if (divisor <= 0) {
                throw std::invalid_argument("divisor must be greater than 0");
            }
        }

        Tensor internal_do_filter(const Tensor &mod_tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            Tensor proximal_tensor = get_tensor(proximity_name_, mod_tensor, transform_type_, ctx);

            // Select the lattice point with this residue closest to the reference.
            Tensor residues = mod_tensor.to(proximal_tensor.options());
            Tensor quotients = at::round((proximal_tensor / lattice_size_ - residues) / divisor_);
            Tensor reconstructed = (quotients * divisor_ + residues) * lattice_size_;
            return reconstructed.to(utils::get_torch_dtype<OriginalValueT>()).to(mod_tensor.device());
        }

        Tensor get_tensor(const std::string &name, const Tensor &inputTensor, MatrixTransformType transform,
                          base::CompressionFilterCtx &ctx) {
            Tensor t = inputTensor;
            if (name.empty() || name == "" || name == "INPUT") {
                t = inputTensor;
            } else {
                t = ctx.get_state<base::SavedMatrixState>(name)->matrix();
            }
            switch (transform) {
                case MatrixTransformType::TRANSPOSE:
                    t = t.t();
                    break;
                case MatrixTransformType::NONE:
                default:
                    break;
            }

            return t;
        }
    };
}
