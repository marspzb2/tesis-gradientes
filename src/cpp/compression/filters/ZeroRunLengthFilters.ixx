module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/masked_select.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/TensorOperators.h>



export module ZeroRunLengthFilters;
import Metrics;
import TensorSerialization;
import SerializationMethods;
import CompressionFilter;
using at::Tensor;
export namespace compression::filters {
    using serialization::SerializationMethod;
    using compression::metrics::Metrics;
    using serialization::VarIntSerialization;
    using serialization::ParallelSerialization;

    template<typename InputT, typename EncodingMetaT>
    class ZeroRunLengthForwardFilter : public base::CompressionFilter<InputT, InputT, base::NoValue, EncodingMetaT> {
    private:
        bool enabled;
        bool metrics_enabled;
        std::unique_ptr<SerializationMethod<InputT, InputT> > serialization_method;

    public:
        explicit ZeroRunLengthForwardFilter(uint8_t encode_size = 8)
            : enabled(std::getenv("ZERO_RLE_DISABLED") != "true"),
              metrics_enabled(true),
              serialization_method(
                  std::make_unique<ParallelSerialization<int64_t, int64_t> >(
                      std::make_unique<VarIntSerialization<InputT, InputT> >(encode_size))) {
        }

        std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                             base::CompressionFilterCtx &compression_ctx) override {
            if (tensor.numel() == 0) {
                return {
                    tensor,
                    at::cat({at::tensor({0, 0, false}, meta_tensor.options()), meta_tensor})
                };
            }
            Tensor nonzero_element_mask = tensor.ne(0);
            //donde cambia de 0 a un valor
            Tensor change_indices = (
                                        nonzero_element_mask.slice(0, 1) !=
                                        nonzero_element_mask.slice(0, 0, -1)
                                    )
                                    .nonzero()
                                    .squeeze(-1) + 1;
            Tensor run_start = at::cat({
                at::tensor({0}, change_indices.options()),
                change_indices
            });
            Tensor run_ends = at::cat({
                change_indices,
                at::tensor({tensor.numel()}, run_start.options()),
            });

            Tensor run_lengths = run_ends - run_start;
            if (metrics_enabled) {
                double total_runs_sum = run_lengths.sum().item<double>();
                Metrics::metrics().number(metrics::NumberTrackingOptions{.name = "zero-run-length"})
                        .register_number(total_runs_sum*1./run_lengths.numel(), false,false)
                        .register_min_max(
                            run_lengths.min().item<double>(),
                            run_lengths.max().item<double>())
                        .add_aggregation(total_runs_sum, run_lengths.numel());
            }
            auto original_run_length = run_lengths.numel();
            run_lengths = serialization_method->serialize(run_lengths, original_run_length);
            auto is_start_nzero = tensor[0].ne(0).to(meta_tensor.options()).reshape({1});

            Tensor non_zero_elements = at::masked_select(tensor, nonzero_element_mask);
            return {
                non_zero_elements,
                at::cat({
                    at::tensor({tensor.numel(), original_run_length, run_lengths.numel()}, meta_tensor.options()),
                    is_start_nzero,
                    run_lengths,
                    meta_tensor
                })
            };
        }
    };

    struct ZeroRunLengthMetadata {
        Tensor run_lengths;
        int64_t encoded_run_length;
        int64_t original_size;
        bool start_with_non_zero;
    };

    template<typename InputT, typename EncodingMetaT>
    class ZeroRunLengthBackwardFilter : public base::CompressionFilter<InputT, InputT, ZeroRunLengthMetadata,
                EncodingMetaT> {
    private:
        bool enabled;
        std::unique_ptr<SerializationMethod<InputT, InputT> > serialization_method;

    public:
        explicit ZeroRunLengthBackwardFilter(uint8_t encode_size = 8)
            : enabled(std::getenv("ZERO_RLE_DISABLED") != "true"),
              serialization_method(
                  std::make_unique<ParallelSerialization<int64_t, int64_t> >(
                      std::make_unique<VarIntSerialization<InputT, InputT> >(encode_size))) {
        }

    public:
        Tensor internal_do_filter(const Tensor &tensor, const std::optional<ZeroRunLengthMetadata> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            if (metadata->original_size == 0) {
                return at::empty({0}, tensor.options());
            }
            int64_t run_count = metadata->run_lengths.numel();
            //std::cout << "Run count: " << metadata->run_lengths.numel() << "Ratio,:: " << run_count * 1. / metadata->
            //        original_size << std::endl;
            Tensor indices = at::arange(run_count, metadata->run_lengths.options());
            Tensor nonzero_run_mask =
                    indices.remainder(2).eq(metadata->start_with_non_zero ? 0 : 1);
            //los pares los ponemos en positivo o negativo dependiendo del start
            Tensor nonzero_element_mask =
                    at::repeat_interleave(nonzero_run_mask, metadata->run_lengths);
            const auto nonzero_element_count =
                    nonzero_element_mask.sum().item<int64_t>();
            Tensor result = at::zeros({metadata->original_size}, tensor.options());
            result.masked_scatter_(nonzero_element_mask, tensor);
            return result;
        }

        Tensor internal_metadata(const Tensor &tensor, const Tensor &meta_tensor,
                                 const std::optional<ZeroRunLengthMetadata> &metadata, base::NoValue *state,
                                 base::CompressionFilterCtx &ctx) override {
            return meta_tensor.slice(0, 4 + metadata->encoded_run_length);
        }

        std::optional<ZeroRunLengthMetadata>
        parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) override {
            int64_t original_size = previousMetadata[0].item<int64_t>();
            int64_t run_count = previousMetadata[1].item<int64_t>();
            int64_t encoded_run_length_count = previousMetadata[2].item<int64_t>();
            bool start_with_non_zero = previousMetadata[3].item<bool>();
            Tensor run_length = serialization_method->deserialize(
                previousMetadata.slice(0, 4, 4 + encoded_run_length_count), run_count);
            return ZeroRunLengthMetadata{
                run_length,
                encoded_run_length_count,
                original_size,
                start_with_non_zero
            };
        };
    };
}
