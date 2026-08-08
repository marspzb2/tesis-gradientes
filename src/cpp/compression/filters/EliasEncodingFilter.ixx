//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
export module EliasEncodingFilter;
import BitUtils;
import EliasCoding;
import CompressionFilter;
import torch_scalar_utils;
using torch::Tensor;

export
{
    namespace compression::filters {
        template<typename InputT, typename OutputT, typename EncodingMetaT>
        class EliasEncodingForwardFilter : public base::CompressionFilter<InputT, OutputT, int64_t, EncodingMetaT> {
        public:
            Tensor internal_do_filter(const Tensor &tensor, const std::optional<long long> &metadata,
                                      base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                assert(tensor.sizes().size()==1);
                Tensor result = torch::zeros_like(tensor,
                                                  torch::TensorOptions(tensor.options()).dtype(
                                                      utils::get_torch_dtype<OutputT>()));

                InputT *data = tensor.to(torch::kCPU).contiguous().data_ptr<InputT>();
                OutputT *result_ptr = result.data_ptr<OutputT>();


                int bit_idx = 0;
                for (int i = 0; i < tensor.numel(); i++) {
                    InputT component_value = data[i];
                    if (component_value >= 0) {
                        component_value += 1;
                    }
                    auto [elias_code,length] = utils::elias_omega_code<
                        OutputT>(static_cast<uint>(abs(component_value)));

                    bit_idx = utils::write_bytes_bige(
                        result_ptr,
                        component_value > 0
                            ? elias_code
                            : elias_code | (1 << length),
                        bit_idx,
                        length + 1);
                }

                return result.slice(
                    0, 0, static_cast<int>(ceil(bit_idx * 1. / utils::get_block_size<OutputT>(result_ptr))));
            }

            std::optional<int64_t> parse_metadata(const Tensor &tensor,const Tensor & meta_tensor) override {
                return std::optional(tensor.numel());
            }

            Tensor internal_metadata(const Tensor &tensor, const Tensor &meta_tensor,
                                     const std::optional<long long> &metadata, base::NoValue *state,
                                     base::CompressionFilterCtx &ctx) override {
                return torch::cat({
                    torch::tensor({
                                      static_cast<EncodingMetaT>(metadata.value())
                                  }, meta_tensor.options()),
                    meta_tensor
                });
            }
        };


        template<typename InputT, typename OutputT, typename EncodingMetaT>
        class EliasDecodingBackwardFilter : public base::CompressionFilter<OutputT, InputT, int64_t, EncodingMetaT> {
        public:
            Tensor internal_do_filter(const Tensor &tensor, const std::optional<int64_t> &metadata,
                                      base::NoValue *state,
                                      base::CompressionFilterCtx &ctx) override {
                Tensor result = torch::zeros(metadata.value(),
                                             torch::TensorOptions().dtype(utils::get_torch_dtype<InputT>()));
                OutputT *data = tensor.to(torch::kCPU).contiguous().data_ptr<OutputT>();
                InputT *components = result.data_ptr<InputT>();
                int bit_idx = 0;

                for (int comp = 0; comp < metadata.value(); comp++) {
                    int sign = utils::bit_at_position(data, bit_idx) == 1 ? -1 : 1;
                    bit_idx++;
                    auto [decoded_component,read_length] = utils::elias_omega_decode(data, bit_idx);
                    components[comp] = static_cast<InputT>(decoded_component) * sign;
                    bit_idx += read_length;
                }
                return result;
            }

            Tensor internal_metadata(const Tensor &tensor, const Tensor &meta_tensor,
                                     const std::optional<int64_t> &metadata, base::NoValue *state,
                                     base::CompressionFilterCtx &ctx) override {
                return meta_tensor.slice(0, 1);
            }

            std::optional<int64_t> parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) override {
                return std::optional<int64_t>(static_cast<int64_t>(previousMetadata[0].item<EncodingMetaT>()));
            }
        };
    }
}
