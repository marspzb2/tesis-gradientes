//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
#include <limits>
#include <spdlog/spdlog.h>
export module NormalizeFilters;
import CompressionFilter;
import torch_scalar_utils;
export namespace compression::filters {
        template<typename ValueT, typename EncodedMetaT>
        class NormalizationForwardFilter : public base::CompressionFilter<ValueT, ValueT, double, EncodedMetaT> {
        public:
            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<double> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                if (metadata.value() < std::numeric_limits<ValueT>::epsilon()) {
                    spdlog::warn("Found zero tensor");
                    return torch::zeros_like(tensor);
                }
                return tensor / metadata.value();
            }


            at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                         const std::optional<double> &metadata, base::NoValue *state,
                                         base::CompressionFilterCtx &ctx) override {
                return torch::cat({
                                      torch::tensor({std::bit_cast<EncodedMetaT>(metadata.value())},
                                                    meta_tensor.options()),
                                      meta_tensor
                                  }, 0);
            }

            std::optional<double>
            parse_metadata(const at::Tensor &tensor, const at::Tensor &previousMetadata) override {
                return tensor.norm().item<double>();
            }
        };


        template<typename ValueT, typename EncodedMetaT>
        class NormalizationBackwardFilter : public base::CompressionFilter<ValueT, ValueT, double, EncodedMetaT> {
        public:
            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<double> &metadata,
                                          base::NoValue *state, base::CompressionFilterCtx &ctx) override {
                if (metadata.value() < std::numeric_limits<ValueT>::epsilon()) {
                    return torch::zeros_like(tensor);
                }
                return tensor * metadata.value();
            }


            at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                         const std::optional<double> &metadata, base::NoValue *state,
                                         base::CompressionFilterCtx &ctx) override {
                return meta_tensor.slice(0, 1); //Remove the first value
            }

            std::optional<double>
            parse_metadata(const at::Tensor &tensor, const at::Tensor &previousMetadata) override {
                return std::bit_cast<double>(previousMetadata[0].item<EncodedMetaT>());
            }
        };

        template<typename ValueT, typename EncodedMetaT>
        class NormalizeByBucketForwardFilter : public base::CompressionFilter<ValueT, ValueT, double, EncodedMetaT> {
        private:
            int bucket_size;
            int p;

        public:
            explicit NormalizeByBucketForwardFilter(int bucket_size,int p)
                : bucket_size(bucket_size),p(p) {
                if (bucket_size <= 0) {
                    throw std::invalid_argument(
                        "El bucket size no es positivo"
                    );
                }

            }

            std::tuple<at::Tensor, at::Tensor> do_filter(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                                         base::CompressionFilterCtx &compression_ctx) override {
                auto tensor_to_normalize = tensor;
                int64_t mod_size = tensor.numel() % bucket_size;
                if (mod_size != 0) {
                    tensor_to_normalize = torch::cat({
                        tensor_to_normalize,
                        torch::zeros({bucket_size - mod_size}, tensor.options())
                    });
                }
                auto tensors_by_bucker = tensor_to_normalize.view({-1, bucket_size});
                auto norms = torch::norm(tensors_by_bucker, p<1000? p:std::numeric_limits<double>::infinity(), {1}, true);
                //Los zero los deja en 1 para dividir
                auto safe_norms = torch::where(
                     norms > 0,
                     norms,
                     torch::ones_like(norms)
                 );
                return std::make_tuple((tensors_by_bucker / safe_norms).flatten(),
                                       torch::cat({
                                           torch::tensor({static_cast<EncodedMetaT>(tensor.numel())},
                                                         meta_tensor.options()),
                                           norms.to(torch::kFloat64)
                                           .contiguous()
                                           .view(utils::get_torch_dtype<EncodedMetaT>())
                                           .flatten(),
                                           meta_tensor
                                       }));
            }
        };

        template<typename ValueT, typename EncodedMetaT>
        class NormalizeByBucketBackwardFilter : public base::CompressionFilter<ValueT, ValueT, double, EncodedMetaT> {
        private:
            int bucket_size;
        public:
            explicit NormalizeByBucketBackwardFilter(int bucket_size)
                : bucket_size(bucket_size) {
                if (bucket_size <= 0) {
                    throw std::invalid_argument(
                        "Bucket size es negativo"
                    );
                }
            }

            std::tuple<at::Tensor, at::Tensor> do_filter(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                                         base::CompressionFilterCtx &compression_ctx) override {
                int64_t mod_size = tensor.numel() % bucket_size;
                if (mod_size != 0) {
                    throw std::runtime_error("The tensor to denormalize doesn't respect the dimension");
                }

                int64_t real_size = static_cast<int64_t>(meta_tensor[0].item<EncodedMetaT>());
                int64_t norm_size = real_size / bucket_size;
                if (real_size % bucket_size != 0) {
                    ++norm_size;
                }
                if (meta_tensor.numel() < 1 + norm_size) {
                    throw std::runtime_error(
                        "La metadata de las norms esta mal"
                    );
                }
                if (tensor.numel() / bucket_size != norm_size) {
                    throw std::runtime_error(
                        "Tensor bucket count does not match metadata"
                    );
                }


                auto tensors_by_bucker = tensor.view({-1, bucket_size});
                auto norms = meta_tensor
                     .slice(0, 1, 1 + norm_size)
                     .contiguous()
                     .view(torch::kFloat64)
                     .to(utils::get_torch_dtype<ValueT>())
                     .view({-1, 1});

                return std::make_tuple((tensors_by_bucker * norms).flatten().slice(0, 0, real_size),
                                       meta_tensor.slice(0, 1 + norm_size));
            }
        };
    template<typename ValueT, typename MetaEncT>
    inline constexpr auto f_normalize_forward = base::FilterCtor<NormalizationForwardFilter<ValueT, MetaEncT>>();
    template<typename ValueT, typename MetaEncT>
    inline constexpr auto f_normalize_backward= base::FilterCtor<NormalizationBackwardFilter<ValueT, MetaEncT>>();

    template<typename ValueT, typename MetaEncT>
    inline constexpr auto f_normalize_by_bucket_forward= base::FilterCtor<NormalizeByBucketForwardFilter<ValueT, MetaEncT>>();

    template<typename ValueT, typename MetaEncT>
    inline constexpr auto f_normalize_by_bucket_backward= base::FilterCtor<NormalizeByBucketBackwardFilter<ValueT, MetaEncT>>();

};

