module;
#include <spdlog/spdlog.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <math.h>
#include <memory>


export module RDAQ;

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

    class RDAQMethod : public BaseCompressionMethod {
        using InputT = float;
        using QuantizedT = int64_t;
        using EncodingT = int64_t;
        using MetaEncT = int64_t;
        using Quantizer = ValueQuantization<InputT, QuantizedT>;
        std::unique_ptr<CompressionFilter<
                InputT, InputT, NoValue, int64_t, int64_t, NoValue>
        >
        filter_chain;
        // k *lattice_size >= 2(lattice_size + delta)
    public:
        RDAQMethod(int bucket_size = 512, int h = 0) {
            if (h < 1)
                h = static_cast<int>(std::exp2(std::ceil(std::log2(1 + iterated_log(bucket_size / 6.)))));
            auto serialization_method = std::make_unique<ParallelSerialization<int64_t, QuantizedT> >(
                std::make_unique<serialization::FixedWidthSerialization<int64_t, QuantizedT> >(1));
            int norm_p{2};
            std::vector<double> scales(h);
            double iterated_e = 1;
            for (int i = 0; i < h; i++) {
                scales[i] = std::sqrt(iterated_e * 6 / bucket_size);
                iterated_e = std::exp(iterated_e);
            }

            auto worker_encoder =
                    f_flatten_tensor_forward<InputT, MetaEncT>()
                    >> f_normalize_by_bucket_forward<InputT, MetaEncT>(bucket_size, norm_p)
                    >> f_save_matrix<InputT, MetaEncT>("OG_REF")
                    >> f_rmq_matrix_init<InputT, int64_t>("ROTATION_MAT", bucket_size)
                    >> f_rmq_rotate_operation_forward<InputT, int64_t>("ROTATION_MAT", std::nullopt, true)
                    >> f_daq_matrix_init<InputT, MetaEncT>("DAQ_MAT", bucket_size, scales)
                    >> f_daq_encode_forward<InputT, QuantizedT, MetaEncT>("DAQ_MAT", h)
                    >> f_serialize_values_forward<QuantizedT, EncodingT, MetaEncT>(serialization_method->clone());
            auto worker_decoder = f_pass_trough<InputT, MetaEncT>();
            auto master_encoder = f_pass_trough<InputT, MetaEncT>();
            auto master_decoder =
                    f_serialize_values_backward<QuantizedT, EncodingT, int64_t>(serialization_method->clone())
                    >> f_daq_matrix_recover<QuantizedT, MetaEncT>("DAQ_MAT_RECV", bucket_size, scales)
                    >> f_daq_decode<InputT, QuantizedT, MetaEncT>("DAQ_MAT_RECV", "OG_REF", h)
                    >> f_normalize_by_bucket_backward<InputT, MetaEncT>(bucket_size)
                    >> f_flatten_tensor_backward<InputT, MetaEncT>();

            using Decoder = typename decltype(master_decoder)::element_type;
            using Reduction = FilterReductionStrategy<Decoder>;
            auto master_reduction_strategy = Reduction::meanReductionStrategy(std::move(master_decoder));
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

        static double iterated_log(double number) {
            if (number <= 1)return 0;
            return 1 + iterated_log(std::log(number));
        }
    };
}
