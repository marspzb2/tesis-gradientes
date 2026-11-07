module;
#include <spdlog/spdlog.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <cmath>


export module DAQ;

import BaseMethod;
import DAQFilters;
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

    class DAQMethod : public BaseCompressionMethod {
        using InputT = float;
        using QuantizedT = int64_t;
        using EncodingT = int64_t;
        using MetaEncT = int64_t;
        using Quantizer = ValueQuantization<InputT, QuantizedT>;
        std::unique_ptr<CompressionFilter<
            InputT, InputT, NoValue, int64_t, int64_t, NoValue> > filter_chain;
        // k *lattice_size >= 2(lattice_size + delta)
    public:
        DAQMethod(int k) {
            if (k < 1)
                throw std::invalid_argument("k must be greater or equal than 1.");
            auto serialization_method = std::make_unique<ParallelSerialization<int64_t, QuantizedT> >(
                std::make_unique<serialization::FixedWidthSerialization<int64_t, QuantizedT> >(1)
            );
            int bucket_size{512};
            int norm_p{211111};
            auto worker_encoder =
                    std::make_unique<FlattenShapeForward<InputT, MetaEncT> >()
                    >> std::make_unique<NormalizeByBucketForwardFilter<InputT, MetaEncT> >(bucket_size, norm_p)
                    >> std::make_unique<SaveMatrixFilter<InputT, MetaEncT> >("OG_REF")
                    >> DAQMatrixInit<InputT, MetaEncT>("DAQ_MAT", bucket_size, k)
                    >> f_daq_encode_forward<InputT, QuantizedT, MetaEncT>("DAQ_MAT", k)
                    >> std::make_unique<FixedSerializationForwardFilter<QuantizedT, EncodingT, MetaEncT> >(
                        serialization_method->clone());
            auto worker_decoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_encoder = std::make_unique<NoOpFilter<InputT> >();
            auto master_decoder =
                    std::make_unique<FixedDeserializationBackwardFilter<QuantizedT, EncodingT, int64_t> >(
                        serialization_method->clone())

                    >> DAQMatrixRecover<QuantizedT,MetaEncT>("DAQ_MAT_RECV",bucket_size,k)
                    >> f_daq_decode<InputT,QuantizedT,MetaEncT>("DAQ_MAT_RECV","OG_REF",k)
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
