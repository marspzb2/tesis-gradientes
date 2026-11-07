module;

#include <algorithm>
#include <concepts>
#include <type_traits>
#include <cassert>
#include <limits>
#include <cmath>

export module BitSelectionUtils;
export namespace compression::utils {
    template<typename T>
    concept Unsigned = std::integral<T> && std::is_unsigned_v<T>;

    template<std::integral ValueT, Unsigned ResultT>
    inline ResultT select_least_significant_bits(ValueT value, uint8_t src_length = sizeof(ValueT) * 8,
                                                 uint8_t src_start = 0) {
        assert(src_start + src_length <= sizeof(ValueT) * 8);
        if (src_length == 0) {
            return ResultT{0};
        }
        using UnsignedValueT = std::make_unsigned_t<ValueT>;

        UnsignedValueT ans = static_cast<UnsignedValueT>(value);
        UnsignedValueT mask = src_length == sizeof(UnsignedValueT) * 8
                                  ? std::numeric_limits<UnsignedValueT>::max()
                                  : (UnsignedValueT{1} << src_length) - 1;
        return static_cast<ResultT>(((ans >> src_start) & mask));
    }


    template<std::integral ValueT, Unsigned ResultT>
    inline ResultT select_most_significant_bits(ValueT value, uint8_t src_length = sizeof(ValueT) * 8,
                                                uint8_t src_start = 0) {
        assert(src_start + src_length <= sizeof(ValueT) * 8);
        if (src_length == 0) {
            return ResultT{0};
        }
        using UnsignedValueT = std::make_unsigned_t<ValueT>;

        UnsignedValueT bits = static_cast<UnsignedValueT>(value); // 1110 1111, start:2, length:4
        UnsignedValueT bits_wout_left_part = bits << src_start; // 10 1111 00,
        UnsignedValueT bits_wout_right_part = bits_wout_left_part >> (sizeof(UnsignedValueT) * 8 - src_length);
        // 10 1111 00, shift: 8 -4 = 4
        return static_cast<ResultT>(bits_wout_right_part); //1011
    }

    template<std::integral ValueT, Unsigned ResultT>
    ResultT select_bits_from_left(const ValueT *values, size_t position, int offset, uint8_t length) {
        ValueT current_value = values[position];
        constexpr uint8_t pointer_size = sizeof(ValueT) * 8;
        uint8_t read_bits = 0;
        ResultT result{0};
        while (read_bits < length) {
            int read_length = std::min(pointer_size - offset, length - read_bits);
            result |= select_most_significant_bits<ValueT, ResultT>(current_value, read_length, offset) << (
                length - read_bits - read_length);
            read_bits += read_length;
            offset += read_length;
            if (offset >= pointer_size) {
                offset %= pointer_size;
                position++;
                if (read_bits < length) {
                    current_value = values[position];
                }
            }
        }
        return result;
    }
}
