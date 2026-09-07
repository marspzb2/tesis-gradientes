module;
#include <algorithm>
#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <concepts>
#include <cassert>
#include <limits>
#include <memory>
#include <type_traits>
#include <bit>
#include <vector>
#include <omp.h>
export module VarIntSerialization;
import torch_scalar_utils;
import BitSelectionUtils;
import BitUtils;
import TensorSerialization;
using torch::Tensor;

export namespace compression::serialization {
    using namespace utils;
    template<std::integral InputT, std::integral OutputT>
    class VarIntSerialization : public SerializationMethod<InputT, OutputT> {
    private:
        using UnsignedInputValue = std::make_unsigned_t<InputT>;
        using SerializedUnsignedValueT = std::make_unsigned_t<OutputT>;
        size_t encode_size;

    public:
        explicit VarIntSerialization(uint8_t encode_size)
            : encode_size(encode_size) {
            if (encode_size < 2)
                throw std::runtime_error("VarIntSerialization::VarIntSerialization: Invalid encode size");
            if (sizeof(SerializedUnsignedValueT) * 8 % encode_size != 0) {
                throw std::runtime_error("VarIntSerialization::VarIntSerialization: Invalid encode size");
            }
            if (encode_size - 1 >= sizeof(UnsignedInputValue) * 8) {
                throw std::runtime_error(
                    "VarIntSerialization::VarIntSerialization: Invalid encode size");
            }
        }

        size_t serialize(InputT *source_pointer, OutputT *result_pointer, size_t quantity) override {
            //1111 0001100 1100110
            //encode_size = 8

            const size_t size_of_output = sizeof(SerializedUnsignedValueT) * 8;
            const size_t payload_bits = encode_size - 1;
            const UnsignedInputValue overflow_value = UnsignedInputValue{1} << payload_bits;

            SerializedUnsignedValueT encoded = 0;
            size_t serialized_bits_in_current_position = 0;
            size_t current_position_in_result = 0;

            for (uint64_t i = 0; i < quantity; i++) {
                // Get the value
                UnsignedInputValue valueToEncode = static_cast<UnsignedInputValue>(source_pointer[i]);
                bool read_value = true;

                //put the bits from the least significant chunk to the most significant
                while (read_value) {
                    //1100110
                    auto encoded_part = select_least_significant_bits<UnsignedInputValue, SerializedUnsignedValueT>(
                        valueToEncode,
                        payload_bits,
                        0);
                    if (valueToEncode >= overflow_value) {
                        encoded_part |= SerializedUnsignedValueT{1} << payload_bits;
                        //11100110
                        valueToEncode = valueToEncode >> payload_bits;
                        read_value = true;
                    } else {
                        read_value = false;
                    }
                    encoded |= encoded_part << (size_of_output - serialized_bits_in_current_position - encode_size);

                    result_pointer[current_position_in_result] = encoded;
                    serialized_bits_in_current_position += encode_size;
                    if (serialized_bits_in_current_position >= size_of_output) {
                        encoded = 0;
                        serialized_bits_in_current_position = 0;
                        ++current_position_in_result;
                    }
                }
            };
            return current_position_in_result + (serialized_bits_in_current_position > 0 ? 1 : 0);
        }

        size_t deserialize(OutputT *serialized_pointer, InputT *deserialized_pointer, size_t quantity) override {
            const size_t size_of_deserialized = sizeof(UnsignedInputValue) * 8;
            const size_t size_of_output = sizeof(SerializedUnsignedValueT) * 8;
            const size_t payload_bits = encode_size - 1;
            const SerializedUnsignedValueT continue_value = SerializedUnsignedValueT{1} << payload_bits;
            const UnsignedInputValue overflow_value = UnsignedInputValue{1} << payload_bits;
            UnsignedInputValue decoded = 0;
            uint current_decoded_offset = 0;
            size_t decoded_value_count = 0;

            for (size_t i = 0; decoded_value_count < quantity; i++) {
                SerializedUnsignedValueT valueToDecode = static_cast<SerializedUnsignedValueT>(serialized_pointer[i]);
                for (size_t j = 0; j < size_of_output / encode_size && decoded_value_count < quantity; j++) {
                    UnsignedInputValue value_part = select_most_significant_bits<SerializedUnsignedValueT,
                        UnsignedInputValue>(
                        valueToDecode, encode_size, j * encode_size);
                    decoded |= (value_part & (overflow_value - 1)) << current_decoded_offset;
                    if (!(continue_value & value_part)) {
                        deserialized_pointer[decoded_value_count] = decoded;
                        ++decoded_value_count;
                        current_decoded_offset = 0;
                        decoded = 0;
                    } else {
                        current_decoded_offset += payload_bits;
                    }
                }
            }
            return decoded_value_count;
        }

        std::unique_ptr<SerializationMethod<InputT, OutputT>> clone() override {
            return std::make_unique<VarIntSerialization<InputT, OutputT>>(*this);
        }
        int64_t initial_size_estimation(const at::Tensor &input, int64_t qty) override {
            return qty*2;
        }
    };
}
