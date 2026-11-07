//
// Created by Marcos Perez on 25/7/26.
//
module;
#include <torch/torch.h>
#include <spdlog/spdlog.h>
export module NoCompressionMethod;
import BaseMethod;
import Network;
import CompressionFilter;
using torch::Tensor;

export namespace compression::methods {

    class NoCompressionMethod:public BaseCompressionMethod{
    public:
         Tensor doMethod(Tensor &input, base::CompressionFilterCtx &filter_ctx) {
            auto work= Network::instance().all_reduce(input);
             work->wait();
             return work->result()[0]/filter_ctx.world_size();
        }
    };
}
