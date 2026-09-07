module;
#include <spdlog/spdlog.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>


export module RMQ;

import BaseMethod;
import all_filters;
import ParallelSerialization;
import EliasSerialization;
import ZeroGapSerialization;
import CompressionFilter;

using at::Tensor;


export namespace compression::methods {
    using namespace compression::base;
    using namespace compression::filters;

    template<typename InputT, typename QuantizedT>
    class RMQMethod : public BaseCompressionMethod {
        using Quantizer = ValueQuantization<InputT, QuantizedT>;
        std::unique_ptr<CompressionFilter<
            InputT, InputT, NoValue, int64_t, int64_t, NoValue> > filter_chain;

    public:
        RMQMethod(double lattice_size, int k) {
            auto worker_encoder =
                    regular_lattice_quantization_filter<InputT, int64_t>(lattice_size, true)
                    >> std::make_unique<ForwardModuloQuantizationFilter<int64_t, QuantizedT> >(k);
            auto worker_decoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_encoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_decoder =
                    std::make_unique<BackwardModuloQuantizationFilter<InputT, QuantizedT> >(k,
                        lattice_size,
                        "ORIGINAL");
            using Decoder = typename decltype(master_decoder)::element_type;
            using Reduction = FilterReductionStrategy<Decoder>;
            auto master_reduction_strategy = Reduction::meanReductionStrategy(std::move(master_decoder));
            //std::make_unique<FilterReductionStrategy<decltype(master_decoder)> >(
            //    master_decoder);
            this->filter_chain = std::make_unique<
                ParameterServerFilter<
                    InputT, int64_t, QuantizedT,
                    int64_t, InputT, int64_t> >(
                std::move(worker_encoder),
                std::move(worker_decoder),
                std::move(master_reduction_strategy),
                std::move(master_encoder)
            );
        }

        Tensor doMethod(Tensor &input, CompressionFilterCtx &filter_ctx) {
            filter_ctx.put_state("ORIGINAL", std::make_unique<SavedMatrixState>("ORIGINAL", input));
            return std::get<0>(filter_chain->do_filter(input, at::zeros({1}, at::TensorOptions()
                                                                        .dtype(at::kLong)
                                                                        .device(input.device())), filter_ctx));
        }
    };
}
