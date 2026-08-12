//
// Created by Marcos Perez on 31/5/26.
//
module;
#include  <torch/torch.h>
export module EliasCoding;
import BitUtils;

namespace compression::utils {
    export template<typename CodeT>
    std::tuple<CodeT, uint> elias_omega_code(uint value) {
        if (value < 0)throw std::invalid_argument("value must be greater than 0.");
        uint og = value;
        CodeT elias_code = 0;
        uint length = 1;
        while (value > 1) {
            int prepend_length = bit_width(value);
            elias_code = elias_code | (value << length);
            value = prepend_length - 1;
            length += prepend_length;
        }
        //std::cout << "Elias for:  " << og << " is " << std::format("{:b}\n", elias_code) << std::endl;
        return std::make_tuple(elias_code, length);
    }

    export template<typename CodeT>
    uint elias_omega_code(CodeT *tensor, uint value, int64_t start_idx) {
        if (value < 0)throw std::invalid_argument("value must be greater than 0.");
        CodeT elias_code = 0;
        uint og = value;
        uint length = 1;
        while (value > 1) {
            int prepend_length = bit_width(value);
            //std::cout << "Bit width of  '" << value << "': " << prepend_length << std::endl;
            elias_code = elias_code | (value << length);
            //std::cout << "current elias of:  " << og << " (" << value << ") is " << std::format("{:b}\n", elias_code) <<
            //        std::endl;
            value = prepend_length - 1;
            length += prepend_length;
        }
        // std::cout << "Elias for:  " << og << " is " << std::format("{:b}\n", elias_code) << std::endl;
        write_bytes_littlee(tensor, elias_code, start_idx, length, sizeof(CodeT));
        return length;
    }

    export template<typename CodeT>
    std::tuple<uint, uint> elias_omega_decode(const CodeT *values, int64_t start_idx) {
        uint value = 1;
        int64_t index = start_idx;
        int block_size = sizeof(CodeT)*8;
        while (bit_at_position<CodeT>(values, index, block_size) != 0) {
            //If the next bit is a "1" then read it plus N more bits, and use that binary number as the new value of N. Go back to Step 2.
            uint oldVal= value;
            std::tie(value, index) = bits_chunk(values, index, value+1, block_size);
            std::cout <<"Reading" <<oldVal << " from index "<<index%block_size<<",value is" << value <<" on " <<std::format("'{:b}'",values[index/block_size])<<std::endl;
            //index++;
        }
        uint read_length = (index - start_idx) + 1;
        std::cout<<std::format("Tensor value: {:b}. Original idx: {:d}.Index inf value: {:d}. Total read: {:d}\n" ,
            values[index/block_size],start_idx,
            start_idx%block_size,
            read_length)<<std::endl;
        std::cout<<"Omega decode: "<<value <<"("<<read_length<<") of "<<std::format("{:b}\n",(values[start_idx/block_size]>>(block_size-start_idx%block_size)) & ((1<<(index-start_idx+1))-1))
        <<std::endl;


        return std::make_tuple(value, read_length);
    }
}
