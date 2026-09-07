module;
#include <torch/torch.h>
#include <cassert>
#include <tuple>
export module ZeroGapSerialization;
import TensorSerialization;
import BitSelectionUtils;
import BitConcatUtils;
import SimpleSerialization;
import EliasSerialization;
export namespace compression::serialization {
    template<typename InputT, typename OutputT>
    class ZeroGapSerialization : public SerializationMethod<InputT, OutputT> {
        using UnsignedInputT = std::make_unsigned_t<InputT>;
        using UnsignedOutputT = std::make_unsigned_t<OutputT>;
        std::unique_ptr<SimpleSerializationMethod<InputT, OutputT> > inner_method_;
        std::unique_ptr<SimpleSerializationMethod<int64_t, OutputT> > zero_serialization_;

    public:
        explicit ZeroGapSerialization(std::unique_ptr<SimpleSerializationMethod<InputT, OutputT> > inner_method,
                                      std::unique_ptr<SimpleSerializationMethod<int64_t, OutputT> > zero_serialization)
            : inner_method_(std::move(inner_method)), zero_serialization_(std::move(zero_serialization)) {
        }

        ZeroGapSerialization(const ZeroGapSerialization &other)
            : inner_method_(other.inner_method_->clone_simple()),
              zero_serialization_(other.zero_serialization_->clone_simple()) {
        }

        size_t deserialize(OutputT *serialized_pointer, InputT *deserialized_pointer, size_t quantity) override {
            size_t remaining = quantity;
            size_t decoded_position = 0;
            size_t offset = 0;
            size_t position = 0;
            int serialized_size = sizeof(OutputT) * 8;
            while (remaining > 0) {
                auto [zero_count,read_bits] = zero_serialization_->deserialize(serialized_pointer, position, offset);
                offset += read_bits;
                position += offset / serialized_size;
                offset %= serialized_size;
                assert(zero_count > 0);
                assert(zero_count - 1 <= remaining);
                remaining -= zero_count - 1;
                for (size_t i = 0; i < zero_count - 1; i++) {
                    deserialized_pointer[decoded_position++] = 0;
                }
                if (remaining > 0) {
                    auto [value,read_bits_value] = inner_method_->deserialize(serialized_pointer, position, offset);
                    --remaining;
                    deserialized_pointer[decoded_position++] = value;

                    offset += read_bits_value;
                    position += offset / serialized_size;
                    offset %= serialized_size;
                }
            }
            return position + (offset > 0 ? 1 : 0);
        }

        size_t serialize(InputT *source_pointer, OutputT *result_pointer, size_t quantity) override {
            size_t zero_count = 1;
            size_t input_position = 0;
            size_t output_position = 0;
            size_t output_offset = 0;

            while (input_position < quantity) {
                while (input_position < quantity && source_pointer[input_position] == 0) {
                    input_position++;
                    zero_count++;
                }
                auto [value,length] = zero_serialization_->serialize(zero_count);
                auto [newPosition,newOffset] = utils::append_right(result_pointer, value, length, output_position,
                                                                   output_offset);
                zero_count = 1;
                output_position = newPosition;
                output_offset = newOffset;
                if (input_position < quantity) {
                    std::tie(value, length) = inner_method_->serialize(source_pointer[input_position]);
                    std::tie(newPosition, newOffset) = utils::append_right(result_pointer, value,
                                                                           length, output_position, output_offset);
                    output_position = newPosition;
                    output_offset = newOffset;
                    input_position++;
                }
            }
            return output_position + (output_offset > 0 ? 1 : 0);
        }


        std::unique_ptr<SerializationMethod<InputT, OutputT> > clone() override {
            return std::make_unique<ZeroGapSerialization<InputT, OutputT> >(*this);
        }

        int64_t initial_size_estimation(const at::Tensor &input, int64_t qty) override {
            return (inner_method_->initial_size_estimation(input, qty) * 110) / 100;
        }
    };
}
