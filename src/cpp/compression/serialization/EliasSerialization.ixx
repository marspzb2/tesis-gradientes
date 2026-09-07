module;
#include <torch/torch.h>
#include <bit>
#include <cstdlib>

export module EliasSerialization;
import  TensorSerialization;
import BitSelectionUtils;
import SimpleSerialization;
export namespace compression::serialization {
    template<typename InputT, typename OutputT>
    class EliasSerialization : public SimpleSerializationMethod<InputT, OutputT> {
        using UnsignedInputT = std::make_unsigned_t<InputT>;
        using UnsignedOutT = std::make_unsigned_t<OutputT>;
        const bool signed_elias;

    public:
        explicit EliasSerialization(bool signed_elias)
            : signed_elias(signed_elias) {
            if (signed_elias && !std::is_signed_v<InputT>) {
                throw std::invalid_argument(
                    "signed Elias requires a signed InputT");
            }

        }

        std::tuple<UnsignedOutT, uint8_t> serialize(const InputT &encode_value) override {
            if (encode_value < 0 && !signed_elias) {
                throw std::runtime_error("serialization error: encode_value must be non-negative");
            } else if (encode_value == 0) {
                throw std::runtime_error("serialization error: encode_value must be non-zero");
            }
            UnsignedInputT value;

            if constexpr (std::is_signed_v<InputT>) {
                value = static_cast<UnsignedInputT>(std::abs(encode_value));
            } else {
                value = encode_value;
            }
            UnsignedOutT elias_code{0};
            uint length = 1;
            while (value > 1) {
                uint prepend_length = std::bit_width(value);
                elias_code = elias_code |  static_cast<UnsignedOutT>(value) << length;;
                value = prepend_length - 1;
                length += prepend_length;
            }
            if (signed_elias) {
                ++length;
            }
            if (signed_elias && encode_value < 0) {
                return {UnsignedOutT{1} << (length-1) | elias_code, length };
            }
            return {elias_code, length};
        }

        std::tuple<InputT, uint8_t>
        deserialize(const OutputT *encoded_values, size_t position, uint8_t offset) override {
            uint8_t block_size = sizeof(OutputT) * 8;
            bool is_negative = false;
            int total_read = 0;
            if (signed_elias) {
                is_negative = encoded_values[position] & UnsignedOutT{1} << (block_size - offset - 1);
                offset++;
                if (offset == block_size ) {
                    offset=0;
                    ++position;
                }
                total_read++;
            }
            UnsignedInputT value{1};
            while (static_cast<UnsignedOutT>(encoded_values[position]) & UnsignedOutT{1} << (block_size - offset - 1)) {
                uint oldVal = value;
                total_read += value + 1;
                if (value + offset + 1 >= block_size) {
                    auto left_value = utils::select_most_significant_bits<OutputT, UnsignedInputT>(
                        encoded_values[position],
                        block_size - offset,
                        offset);
                    ++position;
                    offset = value + 1 - (block_size - offset);
                    auto right_value = utils::select_most_significant_bits<OutputT, UnsignedInputT>(
                        encoded_values[position],
                        offset,
                        0);
                    value = left_value << offset | right_value;
                } else {
                    value = utils::select_most_significant_bits<OutputT, UnsignedInputT>(encoded_values[position],
                        value + 1,
                        offset);
                    offset += oldVal + 1;
                }
            }
            if (is_negative) {
                return {InputT{-1} * value, total_read + 1};
            }
            return {value, total_read + 1};
        }

        std::unique_ptr<SimpleSerializationMethod<InputT, OutputT>> clone_simple() override {
            return std::make_unique<EliasSerialization<InputT, OutputT>>(*this);
        }

        int64_t initial_size_estimation(const at::Tensor &input, int64_t qty) override {
            return qty*2;
        }
    };

}
