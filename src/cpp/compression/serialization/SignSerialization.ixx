module;
#include <ATen/core/Tensor.h>
#include <cassert>
#include <tuple>
export module SignSerialization;
import TensorSerialization;
import SimpleSerialization;

export namespace compression::serialization {
    template<typename InputT, typename OutputT>
    class SignedSerialization : public SimpleSerializationMethod<InputT, OutputT> {
        using InnerSerializationMethod = SimpleSerializationMethod<InputT, OutputT>;
        using UnsignedInputT = std::make_unsigned_t<InputT>;
        using UnsignedOutputT = std::make_unsigned_t<OutputT>;
        std::unique_ptr<InnerSerializationMethod> inner_method_;

    public:
        explicit SignedSerialization(std::unique_ptr<InnerSerializationMethod> inner_method)
            : inner_method_(std::move(inner_method)) {
        }

        SignedSerialization(const SignedSerialization &other)
            : inner_method_(other.inner_method_->clone_simple()) {
        }

        std::tuple<UnsignedOutputT, uint8_t> serialize(const InputT &value) override {
            auto [result_val,count] = inner_method_->serialize(static_cast<UnsignedInputT>(value));
            return {result_val | UnsignedOutputT{value < 0} << count, count + 1};
        }

        std::tuple<InputT, uint8_t> deserialize(const OutputT *value, size_t position, uint8_t offset) override {
            auto current = static_cast<UnsignedOutputT>(value[position]);
            int sign = 1;
            size_t item_size = sizeof(UnsignedOutputT) * 8;
            if (current & 1 << (item_size - offset - 1)) {
                sign = -1;
            }

            offset++;
            if (offset == item_size) {
                position++;
                offset = 0;
            }

            auto [result,count] = inner_method_->deserialize(current, position, offset);

            return {result * sign, count + 1};
        }


        std::unique_ptr<SimpleSerializationMethod<InputT, OutputT>> clone_simple() override {
            return std::make_unique<SignedSerialization>(*this);
        }

        int64_t initial_size_estimation(const at::Tensor &input, int64_t qty) override {
            return (inner_method_->initial_size_estimation(input, qty) * 110) / 100;
        }
    };
}
