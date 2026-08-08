//
// Created by Marcos Perez on 6/6/26.
//

module;
#include <torch/torch.h>

export module FixedQuantizationFilters;
import CompressionFilter;
import torch_scalar_utils;
using torch::Tensor;

export
{
    namespace compression::filters {
        template<typename InputT, typename OutputT>
        class ValueQuantization {
        public:
            virtual ~ValueQuantization() = default;

            virtual OutputT quantize_value(InputT value) =0;

            virtual InputT dequantize_value(OutputT value) =0;
        };

        template<typename InputT, typename OutputT, typename EncMetaT>
        class FixedQuantizationForwardFilter : public base::CompressionFilter<InputT, OutputT, EncMetaT> {
        private:
            std::unique_ptr<ValueQuantization<InputT, OutputT> > quantization_schema;

        public:
            explicit FixedQuantizationForwardFilter(
                std::unique_ptr<ValueQuantization<InputT, OutputT> > quantization_schema)
                : quantization_schema(std::move(quantization_schema)) {
            }

            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<EncMetaT> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                Tensor result = torch::zeros_like(
                    tensor, torch::TensorOptions(tensor.options()).dtype(utils::get_torch_dtype<OutputT>()));
                auto original_tensor_data = tensor.data_ptr<InputT>();
                auto result_tensor_data = result.data_ptr<OutputT>();
                for (int64_t i = 0; i < tensor.numel(); i++) {
                    std::cout<<"Holis"<<quantization_schema->quantize_value(original_tensor_data[i])<<std::endl;
                    result_tensor_data[i] = quantization_schema->quantize_value(original_tensor_data[i]);
                }
                return result;
            }
        };

        template<typename InputT, typename OutputT, typename EncMetaT>
        class FixedQuantizationBackwardFilter : public base::CompressionFilter<OutputT, InputT, EncMetaT> {
        private:
            std::unique_ptr<ValueQuantization<InputT, OutputT> > quantization_schema;

        public:
            explicit FixedQuantizationBackwardFilter(
               std::unique_ptr<ValueQuantization<InputT, OutputT> > quantization_schema)
               : quantization_schema(std::move(quantization_schema)) {
            }
            at::Tensor internal_do_filter(const Tensor &tensor, const std::optional<EncMetaT> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                Tensor result = torch::zeros_like(
                    tensor, torch::TensorOptions(tensor.options()).dtype(utils::get_torch_dtype<InputT>()));
                auto original_tensor_data = tensor.data_ptr<OutputT>();
                auto result_tensor_data = result.data_ptr<InputT>();
                for (int64_t i = 0; i < tensor.numel(); i++) {

                    result_tensor_data[i] = quantization_schema->dequantize_value(original_tensor_data[i]);
                }
                return result;
            }
        };

        template<typename InputT, typename OutputT>
        class RegularQuantization : public ValueQuantization<InputT, OutputT> {
        private:
            InputT lattice_size;
            bool reject_zero; //quantize using middles
        public:
            RegularQuantization(InputT lattice_size, bool reject_zero = false)
                : lattice_size(lattice_size),
                  reject_zero(reject_zero) {
            }

            RegularQuantization(int q, InputT interval_size, bool reject_zero = false)
                : lattice_size(reject_zero
                                   ? static_cast<InputT>(interval_size / (q - 1))
                                   : static_cast<InputT>(1. / interval_size)),
                  reject_zero(reject_zero) {
            }

            //       1            2
            //  |....-.... | .....-....... |
            // 0        t_malla       2 t_malla


            OutputT quantize_value(InputT value) override {
                if (reject_zero) {
                    int sign = value >= 0 ? 1 : -1;
                    if (value == 0)return 1;
                    return sign * ceil((abs(value) * 1. / lattice_size));
                } else {
                    return static_cast<OutputT>(value / this->lattice_size);
                }
            }

            InputT dequantize_value(OutputT value) override {
                if (reject_zero) {
                    int sign = value >= 0 ? 1 : -1;
                    return static_cast<InputT>((value - sign) * this->lattice_size + sign * this->lattice_size / 2.);
                } else {
                    return static_cast<InputT>(value * 1. * this->lattice_size);
                }
            }
        };

        template<typename InputT, typename OutputT>
        class RandomDitheringQuantization : public ValueQuantization<InputT, OutputT> {
        private:
            std::unique_ptr<ValueQuantization<InputT, OutputT> > original_scheme;
            // required to be creciente monotona in the integer lattice
        public:
            explicit RandomDitheringQuantization(std::unique_ptr<ValueQuantization<InputT, OutputT> > original_scheme)
                : original_scheme(std::move(original_scheme)) {
            }

            OutputT quantize_value(InputT value) override {
                //apply dithering according to the distance.
                OutputT inner_quantization = original_scheme->quantize_value(value);
                InputT current_quantization_val = original_scheme->dequantize_value(inner_quantization);
                InputT previous_quantization_val = original_scheme->dequantize_value(inner_quantization - 1);
                InputT next_quantization_val = original_scheme->dequantize_value(inner_quantization + 1);
               /* std::cout<<std::format("Value for {} -> {}({}) -> {}({}) ->{}({})",
                    value,
                    inner_quantization-1,
                    previous_quantization_val,
                    inner_quantization,
                    current_quantization_val,
                    inner_quantization+1,
                    next_quantization_val)<<std::endl;*/
                return dither(value,
                              inner_quantization - 1, previous_quantization_val,
                              inner_quantization, current_quantization_val,
                              inner_quantization + 1, next_quantization_val);
            }

            OutputT dither(InputT real_value,
                OutputT a, InputT val_a,
                OutputT b, InputT val_b,
                OutputT c,InputT val_c) {
                if (val_a > val_b && val_a > val_c) {
                    return dither(real_value, b, val_b, c, val_c, a, val_a);
                }
                if (val_a > val_b) {
                    return dither(real_value, b, val_b, a, val_a, c, val_c);
                }
                if (val_b > val_c) {
                    return dither(real_value, a, val_a, c, val_c, b, val_b);
                }
                OutputT min, max;
                InputT q_min, q_max;

                if (real_value <= val_b) {
                    q_min = a;
                    min = val_a;
                    q_max = b;
                    max = val_b;
                } else {
                    q_min = b;
                    q_max = c;
                    min = val_b;
                    max = val_c;
                }
                //std::cout<<std::format("q_min={}, q_max={},a={},b={},c={}", q_min, q_max,a,b,c)<<std::endl;
                if (torch::rand(1).item<InputT>() <= real_value - min) {
                    return q_min;
                } else {
                    return q_max;
                }
            }


            InputT dequantize_value(OutputT value) override {
                return original_scheme->dequantize_value(value);
            }
        };
    }
}
