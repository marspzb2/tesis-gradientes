module;

#include <concepts>
#include <type_traits>
#include <cassert>
#include <tuple>
#include <stdexcept>
#include <sys/stat.h>

export module BitConcatUtils;
import BitSelectionUtils;
export namespace compression::utils {
    template<std::integral ValueT, std::integral AddValueT>
    inline ValueT concat_lsb_from_right(ValueT value, AddValueT add,
                                        uint8_t dest_offset,
                                        uint8_t src_length = sizeof(AddValueT) * 8,
                                        uint8_t src_start = 0) {
        assert(dest_offset + src_length <= sizeof(ValueT) * 8);
        using UnsignedValueT = std::make_unsigned_t<ValueT>;
        UnsignedValueT result_value = static_cast<UnsignedValueT>(value);
        return static_cast<ValueT>(result_value | select_least_significant_bits<AddValueT, UnsignedValueT>(
                                       add, src_length, src_start) <<
                                   dest_offset);
    }


    template<std::integral ValueT, std::integral AddValueT>
    inline ValueT concat_msb_from_right(ValueT value, AddValueT add,
                                        uint8_t dest_offset,
                                        uint8_t src_length = sizeof(AddValueT) * 8,
                                        uint8_t src_start = 0) {
        assert(dest_offset + src_length <= sizeof(ValueT) * 8);
        using UnsignedValueT = std::make_unsigned_t<ValueT>;
        UnsignedValueT result_value = static_cast<UnsignedValueT>(value);
        return static_cast<ValueT>(result_value | select_most_significant_bits<AddValueT, UnsignedValueT>(
                                       add, src_length, src_start) <<
                                   dest_offset);
    }


    template<std::integral ValueT, std::integral AddValueT>
    inline ValueT concat_lsb_from_left(ValueT value, AddValueT add,
                                       uint8_t dest_offset,
                                       uint8_t src_length = sizeof(AddValueT) * 8,
                                       uint8_t src_start = 0) {
        assert(dest_offset + src_length <= sizeof(ValueT) * 8);
        using UnsignedValueT = std::make_unsigned_t<ValueT>;
        UnsignedValueT result_value = static_cast<UnsignedValueT>(value);
        return static_cast<ValueT>(result_value | select_least_significant_bits<AddValueT, UnsignedValueT>(
                                       add, src_length, src_start) << (
                                       sizeof(UnsignedValueT) * 8 - dest_offset - src_length));
    }


    template<std::integral ValueT, std::integral AddValueT>
    inline ValueT concat_msb_from_left(ValueT value, AddValueT add,
                                       uint8_t dest_offset,
                                       uint8_t src_length = sizeof(AddValueT) * 8,
                                       uint8_t src_start = 0) {
        assert(dest_offset + src_length <= sizeof(ValueT) * 8);
        using UnsignedValueT = std::make_unsigned_t<ValueT>;
        UnsignedValueT result_value = static_cast<UnsignedValueT>(value);
        return static_cast<ValueT>(result_value | select_most_significant_bits<AddValueT, UnsignedValueT>(
                                       add, src_length, src_start) << (
                                       sizeof(UnsignedValueT) * 8 - dest_offset - src_length));
    }

    template<std::integral ValueT, std::integral AddValueT>
    inline std::tuple<size_t, uint8_t> append_right(ValueT *values, AddValueT add, uint8_t source_length,
                                                    size_t position, uint8_t offset) {
        using UnsignedValueT = std::make_unsigned_t<ValueT>;
        using UnsignedAddValueT = std::make_unsigned_t<AddValueT>;

        using WorkingT = std::conditional_t<
            (sizeof(UnsignedValueT) >= sizeof(UnsignedAddValueT)),
            UnsignedValueT,
            UnsignedAddValueT
        >;

        uint8_t cell_size = sizeof(ValueT) * 8;
        if (offset >= cell_size) {
            throw std::runtime_error("append_right: offset too large");
        }
        if (source_length > sizeof(AddValueT) * 8) {
            throw std::runtime_error("append_right: source_length too large");
        }
        if (source_length == 0) {
            return {position, offset};
        }

        UnsignedValueT current_value = static_cast<UnsignedValueT>(values[position]);
        uint8_t current_add_value_msb_idx = source_length;
        auto u_add = static_cast<WorkingT>(add);
        while (current_add_value_msb_idx > 0) {
            u_add = select_least_significant_bits<WorkingT, WorkingT>(
                u_add,
                current_add_value_msb_idx
            );
            // 00 00 00 00 | 11 11 11 11   11 11 11 11   11 11 11 11
            // 11 00 00 00 | 00 00 00 00 | 00 00 00 00
            // msb = 24
            // 00 00 00 00 | 11 11 11 11   11 11 11 11   11 11 11 11 & 11......= 00 00 00 00 11 11 11 11  11 11 11 11  11 11 11 11
            // shift = 18
            //... 00 11 11 11
            //    11 00 00 00
            // msb = 24 - 6 = 18
            // 11   11 11 11 11   11 11 11 11
            // shift = 10
            // 11 11 11 11
            //msb  = 10
            // 11   11 11 11 11
            // shift=  2
            // 11 11 11 11
            // msb = 2
            // 11
            // shift = 0

            uint8_t discard_bits = 0;
            if (current_add_value_msb_idx + offset > cell_size) {
                discard_bits = current_add_value_msb_idx - (cell_size - offset);
            }
            uint8_t added_bits = current_add_value_msb_idx - discard_bits;

            auto value_to_join = u_add >> discard_bits;
            current_value |= value_to_join << (cell_size - offset - added_bits);
            current_add_value_msb_idx -= added_bits;
            offset += added_bits;

            if (offset >= cell_size) {
                values[position] = current_value;
                ++position;
                offset = 0;
                current_value = 0;
            }
        }
        if (offset > 0)
            values[position] = current_value;


        return {position, offset};
    }
}
