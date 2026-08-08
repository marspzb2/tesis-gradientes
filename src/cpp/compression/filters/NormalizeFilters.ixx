//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
export module NormalizeFilters;
import CompressionFilter;
export
{
    namespace compression::filters {
        template<typename ValueT, typename EncodedMetaT>
        class NormalizationForwardFilter : public base::CompressionFilter<ValueT, ValueT,double, EncodedMetaT> {
        public:
            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<double> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                return tensor / metadata.value();
            }


            at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                         const std::optional<double> &metadata, base::NoValue *state,
                                         base::CompressionFilterCtx &ctx) override {
                return torch::cat({
                    torch::tensor({std::bit_cast<EncodedMetaT>(metadata.value())},meta_tensor.options()),
                    meta_tensor}, 0);
            }

            std::optional<double>
            parse_metadata(const at::Tensor &tensor, const at::Tensor &previousMetadata) override {
                return tensor.norm().item<double>();
            }
        };


        template<typename ValueT, typename EncodedMetaT>
        class NormalizationBackwardFilter : public base::CompressionFilter<ValueT, ValueT, double,EncodedMetaT> {
        public:
            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<double> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                return tensor * metadata.value();
            }


            at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                         const std::optional<double> &metadata, base::NoValue *state,
                                         base::CompressionFilterCtx &ctx) override {
                return meta_tensor.slice(0, 1); //Remove the first value
            }

            std::optional<double> parse_metadata(const at::Tensor &tensor, const at::Tensor &previousMetadata) override {
                return std::bit_cast<double>(previousMetadata[0].item<EncodedMetaT>());
            }
        };
    };
}
