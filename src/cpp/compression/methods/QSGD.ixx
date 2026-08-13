module;
#include <memory>
#include <torch/torch.h>
#include <spdlog/spdlog.h>
export module QSGD;
import BaseMethod;
import all_filters;
import CompressionFilter;
using torch::Tensor;
using namespace compression::filters;
using namespace compression::base;
export namespace compression::methods {
    using namespace filters;
    using base::CompressionFilter;

    template<typename InputT, typename QuantizedT>
    class BaseQSGDMethod : public BaseCompressionMethod {
        using Quantizer = ValueQuantization<InputT, QuantizedT>;

    private:
        std::unique_ptr<CompressionFilter<
            InputT, QuantizedT, NoValue, int64_t, int64_t, NoValue> > filter_chain;

    public:
        explicit BaseQSGDMethod(int bucket_size,
                                int norm_p,
                                std::unique_ptr<CompressionFilter<InputT, QuantizedT> > encode_q,
                                std::unique_ptr<CompressionFilter<QuantizedT, InputT> > decode_q) {
            auto qsgd_decoder =
                    std::make_unique<EliasDecodingBackwardFilter<QuantizedT, QuantizedT, int64_t> >()
                    >> std::move(decode_q)
                    //>> std::make_unique<FixedQuantizationBackwardFilter<InputT, QuantizedT, int64_t> >(
                    //    std::make_unique<RandomDitheringQuantization<InputT, QuantizedT> >(std::move(decode_q)))
                    >> std::make_unique<NormalizeByBucketBackwardFilter<InputT, int64_t> >(bucket_size)
                    >> std::make_unique<FlattenShapeBack<InputT, int64_t> >();
            using Decoder = typename decltype(qsgd_decoder)::element_type;
            using Reduction = FilterReductionStrategy<Decoder>;
            auto qsgd_compresion = std::make_unique<FlattenShapeForward<InputT, int64_t> >()
                                   >> std::make_unique<NormalizeByBucketForwardFilter<InputT, int64_t> >(
                                       bucket_size, norm_p)
                                   >> std::move(encode_q)
                                   //>> std::make_unique<FixedQuantizationForwardFilter<InputT, QuantizedT, int64_t> >(
                                   //    std::make_unique<RandomDitheringQuantization<InputT, QuantizedT> >(
                                   //        std::move(encode_q)))
                                   >> std::make_unique<EliasEncodingForwardFilter<QuantizedT, QuantizedT, int64_t> >()
                                   >> std::make_unique<AllGatherFilter<QuantizedT, int64_t> >("Lala",
                                       Reduction::meanReductionStrategy(std::move(qsgd_decoder)));
            ///std::make_unique<FilterReductionStrategy<decltype(qsgd_decoder)::element_type> >(
            //   std::move(qsgd_decoder)));

            this->filter_chain = std::move(qsgd_compresion);
        }

        Tensor doMethod(Tensor &input, CompressionFilterCtx &filter_ctx) {
            return std::get<0>(this->filter_chain->do_filter(input, torch::zeros(
                                                                 {1},
                                                                 torch::TensorOptions()
                                                                 .dtype(torch::kInt64)
                                                                 .device(input.device())), filter_ctx));
        }
    };
    double lattice_size(int q) {
        if (q<=0) {
            throw std::logic_error("lattice_size must be greater than 0");
        }
        return 1./(pow(2,q)-1);
    }

    class DoubleQSGD : public BaseQSGDMethod<double, int64_t> {
    public:
        explicit DoubleQSGD(int q = 8, int bucket_size = 512, int norm_p=2, bool dither = true)
            : BaseQSGDMethod<double, int64_t>(bucket_size,
                                              norm_p,
                                              regular_lattice_quantization_filter<double, int64_t>(lattice_size(q), dither),
                                              regular_lattice_dequantize_filter<double, int64_t>(lattice_size(q))) {
        }
    };

    class FloatQSGD : public BaseQSGDMethod<float, int64_t> {
    public:
        explicit FloatQSGD(int q = 8, int bucket_size = 512, int norm_p=2, bool dither = true)
            : BaseQSGDMethod<float, int64_t>(
                bucket_size,
                norm_p,
                regular_lattice_quantization_filter<float, int64_t>(lattice_size(q), dither),
                regular_lattice_dequantize_filter<float, int64_t>(lattice_size(q))) {
        }
    };
}
