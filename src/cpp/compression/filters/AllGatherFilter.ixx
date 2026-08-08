//
// Created by Marcos Perez on 7/6/26.
//

//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <torch/torch.h>

#include <utility>
export module AllGatherFilter;

import CompressionFilter;
using namespace compression::base;
using torch::Tensor;
using torch::TensorOptions;

export
{
    namespace compression::filters {
        template<typename ValueT, typename EncodingT>
        class AllGatherFilter : public CompressionFilter<ValueT, ValueT, NoValue, EncodingT> {
        private:
            std::string output_save_name;
            std::unique_ptr<ReductionStrategy> reduction_strategy;

        public:
            explicit AllGatherFilter(const std::string &output_save_name,
                                     std::unique_ptr<ReductionStrategy> reduction_strategy)
                : output_save_name(output_save_name),
                  reduction_strategy(std::move(reduction_strategy)) {
            }

            std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &metadata,
                                                 CompressionFilterCtx &ctx) override {
                //if (1 > 0) {
                //    return tensor;
                //}
                auto [work,vectors] = ctx.network().get().all_gather(false, tensor, metadata);
                work->wait();
                auto [reduced_tensor,reduced_meta] = reduction_strategy->reduce(vectors, ctx);


                if (!this->output_save_name.empty()) {
                    this->add_matrix_state(output_save_name, reduced_tensor, ctx, true);
                }

                return std::make_tuple(reduced_tensor, reduced_meta);
            }
        };
    }
}
