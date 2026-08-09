//
// Created by Marcos Perez on 25/7/26.
//
module;
#include <torch/torch.h>
#include <spdlog/spdlog.h>
export module NoCompressionMethod;
import BaseMethod;
import Network;
import all_filters;
import CompressionFilter;
using torch::Tensor;

export namespace compression::methods {
    using namespace filters;

    class NoCompressionMethod:public BaseCompressionMethod{
    public:
         Tensor doMethod(Tensor &input, base::CompressionFilterCtx &filter_ctx) {
            auto work= Network::instance().all_reduce(input)/filter_ctx.world_size();
             work->wait();
             return work->result()[0];
        }
    };
}
