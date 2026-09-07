module;
#include <tuple>
#include <c++/v1/memory>
export module SimpleSerialization;

import TensorSerialization;
import BitSelectionUtils;
namespace compression::serialization {
    /**
     * Va valor a valor tirando transformando directamente.
     * @tparam InputT  La entrada.
     * @tparam OutputT La salida.
     */
    export template<std::integral InputT, std::integral OutputT>
    class SimpleSerializationMethod : public SerializationMethod<InputT, OutputT> {
        using UnsignedOutT = std::make_unsigned_t<OutputT>;
        using UnsignedInputT = std::make_unsigned_t<InputT>;

    public:
        /**
         * Devuelve el valor serializado con su largo.
         * @param value El valor a serializar.
         */
        virtual std::tuple<UnsignedOutT, uint8_t> serialize(const InputT &value) =0;

        virtual std::tuple<InputT, uint8_t> deserialize(const OutputT *value, size_t position, uint8_t offset) =0;

        size_t deserialize(OutputT *serialized_pointer, InputT *deserialized_pointer, size_t quantity) override {
            size_t current_value_cell_idx{0};
            uint8_t current_value_offset{0};
            uint8_t serialized_size_cell = sizeof(UnsignedOutT) * 8;
            for (size_t i = 0; i < quantity; i++) {
                auto [result,read_bits] = this->deserialize(serialized_pointer, current_value_cell_idx,
                                                            current_value_offset);
                deserialized_pointer[i] = result;
                size_t next_offset = (current_value_offset + read_bits);
                current_value_cell_idx += next_offset / serialized_size_cell;
                current_value_offset = next_offset % serialized_size_cell;
            }
            return current_value_cell_idx + (current_value_offset > 0 ? 1 : 0);
        }

        size_t serialize(InputT *source_pointer, OutputT *result_pointer, size_t quantity) override {
            size_t current_output_cell{0};
            UnsignedOutT current_output_value{0};
            size_t current_output_offset{0};
            uint8_t size_of_output_cell = sizeof(UnsignedOutT) * 8;

            for (size_t i = 0; i < quantity; i++) {
                InputT value_to_encode = source_pointer[i];
                auto [encoded_value,encoded_length] = this->serialize(value_to_encode);

                if (encoded_length + current_output_offset > size_of_output_cell) {
                    //esta cortada
                    // offset = 4
                    // length = 6
                    // valor= 11 01 10
                    // current = 11 11 00 00

                    uint8_t left_bit_count = size_of_output_cell - current_output_offset;
                    auto right_bit_count = (encoded_length-left_bit_count);
                    current_output_value |= (encoded_value >> right_bit_count);
                    //current = 11 11 00 00 | 00 00 11 01 = =11 11 11 01
                    result_pointer[current_output_cell] = current_output_value;
                    ++current_output_cell;
                    //append to the end the rest of the bits
                    auto encoded_value_wout_appended = encoded_value & ((UnsignedOutT{1}<<right_bit_count)-1);
                    //11 01 10 & (1 00 -1) = 11 01 10 &(11) = 10


                    current_output_value = encoded_value_wout_appended << (size_of_output_cell - right_bit_count);
                    // offset = 6 - 4 = 2
                    current_output_offset = encoded_length - left_bit_count;
                } else {
                    current_output_value |= encoded_value << size_of_output_cell - current_output_offset -
                            encoded_length;
                    current_output_offset += encoded_length;
                    if (current_output_offset==size_of_output_cell) {
                        result_pointer[current_output_cell++] = current_output_value;
                        current_output_offset=0;
                        current_output_value=0;
                    }
                }
            }
            if (current_output_offset > 0) {
                result_pointer[current_output_cell] = current_output_value;
                return current_output_cell + 1;
            }
            return current_output_cell;
        }
        virtual std::unique_ptr<SimpleSerializationMethod<InputT, OutputT> > clone_simple() =0;

        std::unique_ptr<SerializationMethod<InputT, OutputT> > clone() override {
            return clone_simple();
        }
    };
}
