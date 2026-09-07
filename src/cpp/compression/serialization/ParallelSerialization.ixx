module;
#include <algorithm>
#include <torch/torch.h>
#include <stdexcept>
#include <limits>
#include <memory>
#include <vector>
#include <omp.h>

export module ParallelSerialization;
import TensorSerialization;
import torch_scalar_utils;
using torch::Tensor;

export namespace compression::serialization {
    template<std::integral InputT, std::integral OutputT>
    class ParallelSerialization final : public SerializationMethod<InputT, OutputT> {
    private:
        std::unique_ptr<SerializationMethod<InputT, OutputT> > inner_method_;
        size_t max_chunks_;

        static size_t checked_size(OutputT value, const char *field) {
            if constexpr (std::is_signed_v<OutputT>) {
                if (value < 0) {
                    throw std::runtime_error(
                        std::string("ParallelSerialization: negative ") + field);
                }
            }
            return static_cast<size_t>(value);
        }

    public:
        explicit ParallelSerialization(
            std::unique_ptr<SerializationMethod<InputT, OutputT> > inner_method,
            size_t max_chunks = static_cast<size_t>(omp_get_max_threads()))
            : inner_method_(std::move(inner_method)),
              max_chunks_(max_chunks) {
            if (!inner_method_) {
                throw std::invalid_argument(
                    "ParallelSerialization: inner method is null");
            }
            if (max_chunks_ == 0) {
                throw std::invalid_argument(
                    "ParallelSerialization: max_chunks must be positive");
            }
        }

        ParallelSerialization(const ParallelSerialization &other)
            : inner_method_(other.inner_method_->clone()),
              max_chunks_(other.max_chunks_) {
        }

        Tensor serialize(const Tensor &input, size_t quantity) override {
            if (quantity > static_cast<size_t>(input.numel())) {
                throw std::out_of_range(
                    "ParallelSerialization: quantity exceeds input");
            }

            const auto output_options = torch::TensorOptions()
                    .device(torch::kCPU)
                    .dtype(utils::get_torch_dtype<OutputT>());

            if (quantity == 0) {
                return torch::empty({0}, output_options).to(input.device());
            }

            const Tensor flat_input = input.to(torch::kCPU).contiguous().flatten();
            const size_t chunk_count = std::min(max_chunks_, quantity);
            const size_t base_size = quantity / chunk_count;
            const size_t remainder = quantity % chunk_count;

            std::vector<Tensor> encoded_chunks(chunk_count);
            std::vector<size_t> decoded_sizes(chunk_count);
            std::vector<size_t> encoded_sizes(chunk_count);

#pragma omp parallel for schedule(static)
            for (int64_t chunk = 0; chunk < static_cast<int64_t>(chunk_count); ++chunk) {
                auto inner_method = inner_method_->clone();
                const size_t chunk_index = static_cast<size_t>(chunk);
                const size_t decoded_size = base_size + (chunk_index < remainder ? 1 : 0);
                const size_t start = chunk_index * base_size + std::min(chunk_index, remainder);

                Tensor encoded = inner_method->serialize(
                    flat_input.slice(
                        0,
                        static_cast<int64_t>(start),
                        static_cast<int64_t>(start + decoded_size)),
                    decoded_size);

                decoded_sizes[chunk_index] = decoded_size;
                encoded_sizes[chunk_index] = static_cast<size_t>(encoded.numel());
                encoded_chunks[chunk_index] = std::move(encoded);
            }

            if (chunk_count > static_cast<size_t>(std::numeric_limits<OutputT>::max())) {
                throw std::overflow_error(
                    "ParallelSerialization: chunk count does not fit in OutputT");
            }

            Tensor header = torch::zeros(
                {static_cast<int64_t>(1 + 2 * chunk_count)},
                output_options);
            OutputT *header_data = header.data_ptr<OutputT>();
            header_data[0] = static_cast<OutputT>(chunk_count);

            for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
                if (decoded_sizes[chunk] >
                    static_cast<size_t>(std::numeric_limits<OutputT>::max()) ||
                    encoded_sizes[chunk] >
                    static_cast<size_t>(std::numeric_limits<OutputT>::max())) {
                    throw std::overflow_error(
                        "ParallelSerialization: chunk size does not fit in OutputT");
                }

                header_data[1 + 2 * chunk] = static_cast<OutputT>(decoded_sizes[chunk]);
                header_data[2 + 2 * chunk] = static_cast<OutputT>(encoded_sizes[chunk]);
            }

            encoded_chunks.insert(encoded_chunks.begin(), header);
            return torch::cat(encoded_chunks).to(input.device());
        }

        Tensor deserialize(const Tensor &input, int size) override {
            if (size < 0) {
                throw std::invalid_argument(
                    "ParallelSerialization: negative output size");
            }
            if (size == 0) {
                return torch::empty(
                    {0},
                    torch::TensorOptions()
                    .dtype(utils::get_torch_dtype<InputT>())
                    .device(input.device()));
            }

            const Tensor flat_input = input.to(torch::kCPU).contiguous().flatten();
            if (flat_input.numel() == 0) {
                throw std::runtime_error(
                    "ParallelSerialization: missing header");
            }

            const OutputT *data = flat_input.data_ptr<OutputT>();
            const size_t chunk_count = checked_size(data[0], "chunk count");
            const size_t header_size = 1 + 2 * chunk_count;

            if (header_size > static_cast<size_t>(flat_input.numel())) {
                throw std::runtime_error(
                    "ParallelSerialization: truncated header");
            }

            std::vector<size_t> decoded_sizes(chunk_count);
            std::vector<size_t> encoded_sizes(chunk_count);
            std::vector<size_t> payload_offsets(chunk_count);
            size_t decoded_total = 0;
            size_t encoded_total = 0;

            for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
                decoded_sizes[chunk] = checked_size(data[1 + 2 * chunk], "decoded size");
                encoded_sizes[chunk] = checked_size(data[2 + 2 * chunk], "encoded size");
                payload_offsets[chunk] = header_size + encoded_total;
                decoded_total += decoded_sizes[chunk];
                encoded_total += encoded_sizes[chunk];
            }

            if (decoded_total != static_cast<size_t>(size) ||
                header_size + encoded_total > static_cast<size_t>(flat_input.numel())) {
                throw std::runtime_error(
                    "ParallelSerialization: invalid chunk table");
            }

            std::vector<Tensor> decoded_chunks(chunk_count);

#pragma omp parallel for schedule(static)
            for (int64_t chunk = 0; chunk < static_cast<int64_t>(chunk_count); ++chunk) {
                auto inner_method = inner_method_->clone();
                const size_t chunk_index = static_cast<size_t>(chunk);
                const size_t offset = payload_offsets[chunk_index];
                decoded_chunks[chunk_index] = inner_method->deserialize(
                    flat_input.slice(
                        0,
                        static_cast<int64_t>(offset),
                        static_cast<int64_t>(offset + encoded_sizes[chunk_index])),
                    static_cast<int>(decoded_sizes[chunk_index]));
            }

            return torch::cat(decoded_chunks).to(input.device());
        }

        size_t serialize(
            InputT *source_pointer,
            OutputT *result_pointer,
            size_t quantity) override {
            return inner_method_->serialize(source_pointer, result_pointer, quantity);
        }

        size_t deserialize(
            OutputT *serialized_pointer,
            InputT *deserialized_pointer,
            size_t quantity) override {
            return inner_method_->deserialize(
                serialized_pointer,
                deserialized_pointer,
                quantity);
        }

        std::unique_ptr<SerializationMethod<InputT, OutputT> > clone() override {
            return std::make_unique<ParallelSerialization<InputT, OutputT> >(*this);
        }
    };
}
