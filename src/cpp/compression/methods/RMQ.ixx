module;
#include <spdlog/spdlog.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <cmath>


export module RMQ;

import BaseMethod;
import all_filters;
import ParallelSerialization;
import SignSerialization;
import FixedSerialization;
import CompressionFilter;
import RMQMatrixInitFilter;
import MatrixMultiplyFilter;

using at::Tensor;


export namespace compression::methods {
    using namespace compression::base;
    using namespace compression::filters;

    class RMQMethod : public BaseCompressionMethod {
        using InputT = float;
        using QuantizedT = int64_t;
        using ModuloT = int64_t;
        using Quantizer = ValueQuantization<InputT, QuantizedT>;
        std::unique_ptr<CompressionFilter<
            InputT, InputT, NoValue, int64_t, int64_t, NoValue> > filter_chain;
        // k *lattice_size >= 2(lattice_size + delta)
    public:
        RMQMethod(double lattice_size, int k) {
            if (k <= 1)
                throw std::invalid_argument("k must be greater than 1.");
            auto serialization_method = std::make_unique<serialization::ParallelSerialization<int64_t, QuantizedT> >(
                std::make_unique<serialization::FixedWidthSerialization<int64_t, QuantizedT> >(
                    static_cast<uint8_t>(std::ceil(std::log2(k))))
            );
            int bucket_size{512};
            int norm_p{2};
            auto worker_encoder =
                    std::make_unique<FlattenShapeForward<InputT, int64_t> >()
                    >> std::make_unique<NormalizeByBucketForwardFilter<InputT, int64_t> >(bucket_size, norm_p)
                    >> std::make_unique<SaveMatrixFilter<InputT, int64_t> >("OG_REF")
                    >> f_rmq_matrix_init<InputT, int64_t>("ROTATION_MAT", bucket_size)
                    >> f_rmq_rotate_operation_forward<InputT, int64_t>( "ROTATION_MAT", std::nullopt, true)
                    >> regular_lattice_quantization_filter<InputT, ModuloT>(lattice_size, true)
                    >> std::make_unique<ForwardModuloQuantizationFilter<ModuloT, ModuloT> >(k)
                    >> std::make_unique<FixedSerializationForwardFilter<ModuloT, QuantizedT, int64_t> >(
                        serialization_method->clone());
            auto worker_decoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_encoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_decoder =
                    std::make_unique<FixedDeserializationBackwardFilter<ModuloT, QuantizedT,
                        int64_t> >(
                        serialization_method->clone())
                    >> f_rmq_matrix_recover<QuantizedT, int64_t>("ROTATION_MAT", bucket_size)
                    >> f_rmq_rotate_operation_forward<QuantizedT, int64_t>("ROTATION_MAT",
                                                           "OG_REF",
                                                           false,
                                                           "RMQ_REFERENCE")
                    >> std::make_unique<BackwardModuloQuantizationFilter<InputT, QuantizedT> >(
                        k, lattice_size, "RMQ_REFERENCE")
                    >> f_rmq_rotate_operation_backward<InputT, int64_t>("ROTATION_MAT", true)
                    >> std::make_unique<NormalizeByBucketBackwardFilter<InputT, int64_t> >(bucket_size)
                    >> std::make_unique<FlattenShapeBack<InputT, int64_t> >();

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
            return std::get<0>(filter_chain->do_filter(input, at::zeros({1}, at::TensorOptions()
                                                                        .dtype(at::kLong)
                                                                        .device(input.device())), filter_ctx));
        }
    };
}
