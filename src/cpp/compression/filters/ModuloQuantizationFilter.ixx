module;
#include <concepts>
#include <ATen/core/Tensor.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/eye.h>
#include <ATen/ops/floor.h>
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
        QuantizationT divisor_;

    public:
        explicit ForwardModuloQuantizationFilter(QuantizationT divisor)
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
        QuantizationT divisor_;

        double lattice_size_;
        std::string proximity_name_;
        MatrixTransformType transform_type_;

    public:
        explicit BackwardModuloQuantizationFilter(QuantizationT divisor,
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

            // tengo = 11.5, 11/epsilon =22 -> (10*2+2) | (10*3+2)
            // minimo de (z*k + 1) * epsilon = ...,0.5, 5.5, 10.5, 15.5,.
            //                                  z=  0,  1   , 2 ,   3
            //va a ser 2

            //tengo = 9, 9/epsilon = 18 -> (10*1 + 8) | (10*2 +8)
            // minimo de (z*k + 1) * epsilon = ...,0.5, 5.5, 10.5, 15.5,.
            //                                  z=  0,  1   , 2 ,   3
            // va a ser 2

            //tengo =5 -> 5/epsilon = 10 esta lejos no importa
            // minimo de (z*k + 1) * epsilon = ...,0.5, 5.5, 10.5, 15.5,.
            //                                  z=  0,  1   , 2 ,   3
            // va a ser 1


            Tensor sign= at::sign(proximal_tensor);
            proximal_tensor = at::abs(sign);
            Tensor proximal_quotients = at::floor(at::div(proximal_tensor, divisor_ * lattice_size_));

            Tensor reconstruction_error = ((proximal_quotients * divisor_ + mod_tensor) - at::div(
                                               proximal_tensor, lattice_size_)) / divisor_;

            Tensor quotient_reconstruction = proximal_quotients * divisor_ + mod_tensor
                                             + divisor_ * (reconstruction_error > 0.5).to(proximal_quotients.dtype());


            return (quotient_reconstruction * lattice_size_).to(utils::get_torch_dtype<OriginalValueT>()).to(
                mod_tensor.device());
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
