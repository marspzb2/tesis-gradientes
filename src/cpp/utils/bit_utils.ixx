//
// Created by Marcos Perez on 31/5/26.
//
module;
#include <torch/torch.h>

export module BitUtils;
using index_t = int64_t;
using offset_t = int64_t;
using block_idx_t = int64_t;
using bit_t = bool;

namespace compression::utils {
    export template<typename T>
    uint get_block_size(T *values) {
        return sizeof(T) * 8;
    }

    export std::tuple<block_idx_t, offset_t> current_position(index_t current_idx, uint block_size) {
        return std::make_tuple(current_idx / block_size, current_idx % block_size);
    }

    export template<typename ValueT>
    bit_t bit_at_position(const ValueT *tensor, index_t idx,
                          uint block_size = 0) {
        if (block_size == 0)
            block_size = get_block_size(tensor);
        int block, block_offset;
        std::tie(block, block_offset) = current_position(idx, block_size);
        return (tensor[block] >> (block_size - block_offset - 1)) & 1;
    }

    export template<typename ValueT>
    std::tuple<uint64_t, index_t> bits_chunk(const ValueT *tensor,
                                             index_t current_idx,
                                             int length,
                                             uint block_size = 0) {
        if (block_size == 0)
            block_size = get_block_size(tensor);

        auto [start_block, start_block_offset] = current_position(current_idx, block_size);
        auto [end_block, end_block_offset] = current_position(current_idx + length, block_size);

        if (end_block == start_block) {
            return std::make_tuple(
                (tensor[start_block] >> (block_size - end_block_offset)) & (
                    (1 << (end_block_offset - start_block_offset)) - 1),
                current_idx + length);
        } else {
            uint64_t start_mask = (1 << (block_size - start_block_offset)) - 1;
            uint64_t current_chunk = tensor[start_block] & start_mask;
            for (int blk = start_block + 1; blk < end_block - 1; blk++) {
                current_chunk <<= block_size;
                current_chunk |= tensor[blk];
            }
            current_chunk <<= end_block_offset;
            current_chunk |= tensor[end_block] >> (block_size - end_block_offset);
            return std::make_tuple(current_chunk, current_idx + length);
        }
    }

    export template<typename T>
    T get_trailing_bits_little_e(const T value, uint quantity) {
        return value & ((1 << quantity) - 1);
    }

    export template<typename T>
    T get_heading_bits_little_e(const T value, uint quantity) {
        return std::bit_cast<T>(std::bit_cast<uint64_t>(value) >> (64 - quantity));
    }


    export template<typename ValueT>
    index_t write_bytes_bige(ValueT *tensor, uint64_t value, index_t current_position, uint quantity,
                             uint block_size = 0) {
        if (block_size == 0)
            block_size = get_block_size(tensor);
        uint64_t blk = current_position / block_size;
        uint blk_offset = current_position % block_size;

        // block 10
        //offset 1
        if (blk_offset + quantity < block_size) {

            auto old_val=tensor[blk];

            tensor[blk] |= get_trailing_bits_little_e(value, quantity) << (block_size - blk_offset - quantity );
            std::cout << std::format("Writted {:b}({}) on index {} ({}/{}) of value '{:b}'...'{:b}'\n",
                                     value,
                                     quantity,
                                     blk_offset,
                                     blk,
                                     current_position,
                                     old_val,
                                     tensor[blk]

            );
        } else {
            int first_block_write = block_size - blk_offset;
            int last_block_write = quantity - first_block_write;
            auto old_val_a=tensor[blk];
            auto old_val_b=tensor[blk+1];
            tensor[blk] |= value >> last_block_write;
            tensor[blk + 1] |= get_trailing_bits_little_e(value, last_block_write) << (
                block_size - last_block_write );

            std::cout << std::format("Writted {:b}({}/{:b}) on index {} ({}/{}) of value '{:b}'...'{:b}' and '{:b}'....'{:b}'\n",
                                    value,
                                    first_block_write,
                                    value>>last_block_write,
                                    blk_offset,
                                    blk,
                                    current_position,
                                    old_val_a,
                                    tensor[blk],
                                    old_val_b,
                                    tensor[blk+1]

           );
        }
        return current_position + quantity;
    }

    export template<typename ValueT>
    index_t write_bytes_littlee(ValueT *tensor, uint64_t value, index_t current_position, uint quantity,
                                uint block_size = 0) {
        if (block_size == 0)
            block_size = get_block_size(tensor);
        uint64_t blk = current_position / block_size;
        uint blk_offset = current_position % block_size;
        tensor[blk] |= (value & ((1 << (quantity)) - 1)) << (block_size - blk_offset - 1); //101000, block_offset =0
        return current_position + quantity;
    }

    export uint bit_width(uint64_t value) {
        int width = 0;
        while (value != 0) {
            value >>= 1;
            width++;
        }
        return width;
    }
}
