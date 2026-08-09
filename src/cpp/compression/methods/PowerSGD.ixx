//
// Created by Marcos Perez on 25/7/26.
//
module;
#include <torch/torch.h>
#include <spdlog/spdlog.h>
export module PowerSGD;
import BaseMethod;
import all_filters;
import CompressionFilter;
using torch::Tensor;
using namespace compression::filters;
using namespace compression::base;
export namespace compression::methods {
    using namespace filters;
    using base::CompressionFilter;

    template<typename InputT>
    class PowerSGD : public BaseCompressionMethod {
    private:
        const uint rank;
        std::unique_ptr<CompressionFilter<InputT> > filter_chain;
        std::unique_ptr<CompressionFilter<InputT> > transfer_chain;

    public:
        explicit PowerSGD(uint rank)
            : rank(rank) {
            auto chain =
                    MatrixInitializationFilter<InputT, int64_t>::initialize_uniform_matrix("Q", {-2, rank})
                    >> MatOperationFilter<InputT, int64_t>::mat_dot("ORIGINAL", "Q", "P")
                    >> std::make_unique<AllReduceFilter<InputT, int64_t> >("P")
                    >> std::make_unique<OrthogonalizationFilter<InputT, int64_t> >("P_hat")
                    >> MatOperationFilter<InputT, int64_t>::mat_dot("ORIGINAL", "P_hat", "Q",
                                                                    MatrixTransformType::TRANSPOSE)
                    >> std::make_unique<AllReduceFilter<InputT, int64_t> >("Q")
                    //decompress
                    >> MatOperationFilter<InputT, int64_t>::mat_dot("P_hat", "Q", "", MatrixTransformType::NONE,
                                                                    MatrixTransformType::TRANSPOSE);
            this->filter_chain = std::move(chain);
            this->transfer_chain =
                    std::make_unique<AllReduceFilter<InputT, int64_t> >("VEC");
        }

        Tensor doMethod(Tensor &input, CompressionFilterCtx &filter_ctx) {
            filter_ctx.put_state("ORIGINAL", std::make_unique<SavedMatrixState>("ORIGINAL", input));
            if (input.dim() <= 1) {
                spdlog::debug("(POWERSGD) Will run for 1D vector: {}", input.size(0));

                return std::get<0>(this->transfer_chain->do_filter(input, torch::zeros({1}), filter_ctx));
            }
            spdlog::debug("(POWERSGD) Will run for ({})D vector: {}", input.dim(), input.size(0), input.size(1));
            return std::get<0>(this->filter_chain->do_filter(input, torch::zeros({1}), filter_ctx));
        }
    };

    class DoublePowerSGD : public PowerSGD<double> {
    public:
        using PowerSGD<double>::PowerSGD;

        explicit DoublePowerSGD(uint rank)
            : PowerSGD(rank) {
        }
    };

    class FloatPowerSGD : public PowerSGD<float> {
    public:
        using PowerSGD<float>::PowerSGD;

        explicit FloatPowerSGD(uint rank)
            : PowerSGD(rank) {
        }
    };
}
