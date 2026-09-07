module;
#include <torch/torch.h>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <c10/core/ScalarType.h>
#define DEFAULT_BUCKET_SIZE 8192

export module ParallelEliasFilters;
import BitUtils;
import EliasCoding;
import CompressionFilter;
import torch_scalar_utils;
using torch::Tensor;
export namespace compression::filters {
    template<typename InputT, typename OutputT, typename EncodingMetaT>
    class ParallelEliasEncodingForwardFilter : public base::CompressionFilter<InputT, OutputT, int64_t, EncodingMetaT> {
    private:
        int64_t bucket_size = DEFAULT_BUCKET_SIZE;

    public:

        std::tuple<Tensor,Tensor> do_filter(const Tensor &tensor, const Tensor & metadata,base::CompressionFilterCtx &ctx) override {
            assert(tensor.sizes().size() == 1);
            auto output_dtype = utils::get_torch_dtype<OutputT>();
            auto meta_dtype = utils::get_torch_dtype<EncodingMetaT>();
            if (tensor.numel() == 0) {
                return std::make_tuple(tensor.to(output_dtype),
                    torch::cat({
                        torch::tensor({0,0}).to(metadata.device()).view(meta_dtype),
                        metadata
                        }));
            }

            auto tensor_cpu = tensor.to(torch::kCPU).contiguous();

            int64_t bucket_count = tensor.numel() / bucket_size;
            if (tensor.numel() % bucket_size != 0) {
                ++bucket_count;
            }
            std::vector<Tensor> encoded_buckets(bucket_count);
            std::vector<int64_t> real_bucket_lengths(bucket_count);

            // codificar elias en paralelo
#pragma omp parallel for schedule(static)
            for (int64_t bucket_idx = 0; bucket_idx < bucket_count; bucket_idx++) {
                int64_t bit_idx = 0;
                Tensor result_of_bucket = torch::zeros((bucket_size*3)/2, //no va a ser mas que 1.5 la cantidad renderizada
                                             torch::TensorOptions(tensor.options())
                                             .dtype(output_dtype)
                                             .device(torch::kCPU));
                encoded_buckets[bucket_idx] = result_of_bucket;
                OutputT *result_ptr = result_of_bucket.data_ptr<OutputT>();

                InputT *data = tensor_cpu[bucket_idx*bucket_size].data_ptr<InputT>();
                int total_length = 0;
                for (int i = 0; i < std::min(bucket_size,tensor_cpu.numel()-bucket_idx*bucket_size); i++) {
                    InputT component_value = data[i];
                    if (component_value >= 0) {
                        component_value += 1;
                    }
                    auto [elias_code,length] = utils::elias_omega_code<
                        OutputT>(static_cast<uint64_t>(abs(component_value)));

                    bit_idx = utils::write_bytes_bige(
                        result_ptr,
                        component_value > 0
                            ? elias_code
                            : elias_code | (1 << length),
                        bit_idx,
                        length + 1);
                    total_length += length;
                    if ((bit_idx+1)/8 > c10::elementSize(output_dtype)* result_of_bucket.numel()-10) {
                        spdlog::error("Overflow esta a punto de pasar");

                    }
                }
                auto bit_size_of_encoded_type = (c10::elementSize(output_dtype)*8);
                int real_bucket_size = bit_idx/ bit_size_of_encoded_type;
                if (bit_idx % bit_size_of_encoded_type!=0) {
                    ++real_bucket_size;
                }
                encoded_buckets[bucket_idx] = result_of_bucket.slice(0,0, real_bucket_size);
                real_bucket_lengths[bucket_idx] = real_bucket_size;
            }

            return std::make_tuple(
                torch::cat(encoded_buckets).to(tensor.device()),
                torch::cat({
                    torch::tensor({tensor.numel(),bucket_count}).to(metadata.device()).view(meta_dtype),
                    torch::tensor(real_bucket_lengths).to(metadata.device()).view(meta_dtype),
                    metadata
                })
                );
        }

    };


    using BackwardMetadata = std::tuple<int64_t, std::vector<int64_t>>;
    template<typename InputT, typename OutputT, typename EncodingMetaT>
    class
            ParallelEliasDecodingBackwardFilter : public base::CompressionFilter<OutputT, InputT, BackwardMetadata,
                EncodingMetaT> {
    private:
        int64_t b_size{DEFAULT_BUCKET_SIZE};
    public:
        Tensor internal_do_filter(const Tensor &tensor, const std::optional<BackwardMetadata> &metadata,
                                  base::NoValue *state,
                                  base::CompressionFilterCtx &ctx) override {
            int64_t total_components = std::get<0>(metadata.value());
            const std::vector<int64_t> &bucket_sizes = std::get<1>(metadata.value());
            if (total_components == 0)
                return torch::empty(0).to(tensor.device());
            int bucket_count= bucket_sizes.size();
            std::vector<Tensor> results(bucket_count);
            auto tensor_cpu=tensor.to(torch::kCPU).contiguous();
            std::vector<int64_t> offsets(bucket_count);
            for (int bucket_idx = 0; bucket_idx < bucket_count; bucket_idx++) {
                if (bucket_idx==0)
                    offsets[bucket_idx] =0;
                else
                    offsets[bucket_idx] = offsets[bucket_idx-1]+bucket_sizes[bucket_idx-1];
            }
#pragma omp parallel for schedule(static)
            for (int bucket = 0; bucket < bucket_count; bucket++) {
                int64_t bucket_size = std::min(b_size, total_components - bucket * b_size);

                Tensor result = torch::zeros(bucket_size,
                                             torch::TensorOptions().dtype(utils::get_torch_dtype<InputT>()));

                OutputT *data = tensor_cpu[offsets[bucket]].data_ptr<OutputT>();
                InputT *components = result.data_ptr<InputT>();
                int bit_idx = 0;

                for (int comp = 0; comp < bucket_size; comp++) {
                    int sign = utils::bit_at_position(data, bit_idx) == 1 ? -1 : 1;
                    bit_idx++;
                    auto [decoded_component,read_length] = utils::elias_omega_decode(data, bit_idx);
                    if (sign > 0) {
                        components[comp] = static_cast<InputT>(decoded_component - 1) * sign;
                    } else {
                        components[comp] = static_cast<InputT>(decoded_component) * sign;
                    }
                    bit_idx += read_length;
                }
                results[bucket] = result;
            }

            return torch::cat(results).to(tensor.device());
        }

        Tensor internal_metadata(const Tensor &tensor, const Tensor &meta_tensor,
                                 const std::optional<BackwardMetadata> &metadata, base::NoValue *state,
                                 base::CompressionFilterCtx &ctx) override {
            return meta_tensor.slice(0, 2+std::get<1>(metadata.value()).size());
        }

        std::optional<BackwardMetadata> parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) override {
            auto total_comps=previousMetadata[0].view(torch::kInt64).item<int64_t>();
            auto bucket_count = previousMetadata[1].view(torch::kInt64).item<int64_t>();
            std::vector<int64_t> components(bucket_count);
            for (int comp = 0; comp < bucket_count; comp++) {
                components[comp] = previousMetadata[2+comp].view(torch::kInt64).item<int64_t>();
            }
            return std::optional{std::make_tuple(total_comps,components)};
        }
    };
}
