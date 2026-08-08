//
// Created by Marcos Perez on 13/7/26.
//
module;
#include <torch/torch.h>

export module CompresssionMethod;
using Tensor = torch::Tensor;

export namespace compression::base {
    template<
        typename InputT,
        typename OutputT=InputT>
    class CompressionMethod {
    public:
        virtual ~CompressionMethod() = default;

        virtual std::tuple<Tensor, Tensor> compress(Tensor &gradients) =0;

        virtual Tensor decompress(Tensor &gradients, Tensor &metadata) =0;
    };
}
