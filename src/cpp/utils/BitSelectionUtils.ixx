module;

#include <concepts>
#include <type_traits>
#include <cassert>
#include <limits>

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


}
