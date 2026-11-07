module;
#include <ATen/core/Tensor.h>
#include <cassert>
#include <tuple>
export module FixedSerialization;
import TensorSerialization;
import SimpleSerialization;
import BitSelectionUtils;

export namespace compression::serialization {
    template<typename InputT, typename OutputT>
    class FixedWidthSerialization : public SimpleSerializationMethod<InputT, OutputT> {
        using UnsignedInputT = std::make_unsigned_t<InputT>;
        using UnsignedOutputT = std::make_unsigned_t<OutputT>;
        uint8_t width_;

    public:
        explicit FixedWidthSerialization(uint8_t width) : width_(width) {
        }

        FixedWidthSerialization(const FixedWidthSerialization &other)
            : width_(other.width_) {
        }

        std::tuple<UnsignedOutputT, uint8_t> serialize(const InputT &value) override {
            if (value >= (UnsignedOutputT{1} << width_)) {
                throw std::runtime_error("FixedSerialization::FixedSerialization: value is too large");
            }
            return {std::bit_cast<UnsignedOutputT>(value), width_};
        }

        std::tuple<InputT, uint8_t> deserialize(const OutputT *value, size_t position, uint8_t offset) override {
            return {
                std::bit_cast<InputT>(
                    utils::select_bits_from_left<OutputT, UnsignedOutputT>(value, position, offset, width_)),
                width_
            };
        }


        std::unique_ptr<SimpleSerializationMethod<InputT, OutputT> > clone_simple() override {
            return std::make_unique<FixedWidthSerialization>(*this);
        }

        int64_t initial_size_estimation(const at::Tensor &input, int64_t qty) override {
            return qty * (width_ / 64 + 1);
        }
    };
}
