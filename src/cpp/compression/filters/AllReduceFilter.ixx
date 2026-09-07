//
// Created by Marcos Perez on 7/6/26.
//

//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/div.h>
#include <c10/core/TensorOptions.h>


#include <utility>
export module AllReduceFilter;

import CompressionFilter;
using namespace compression::base;
using at::Tensor;
using at::TensorOptions;

export
{
    namespace compression::filters {
        template<typename ValueT, typename EncodingT>
        class AllReduceFilter :public CompressionFilter<ValueT, ValueT, NoValue, EncodingT> {
        private:
            std::string output_save_name;
            bool average_results;

        public:
            explicit AllReduceFilter(const std::string &  output_save_name)
                : output_save_name(output_save_name),average_results(true) {
            }

            Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<NoValue> &metadata,
                                      NoValue *state, CompressionFilterCtx &ctx) override {

                assert(ctx.network() != nullptr);
                assert(ctx.network().get().get_process_group() != nullptr);

                auto work = ctx.network().get().all_reduce(tensor);
                work->wait();
                Tensor reduced = work->result()[0];
                if (average_results) {
                    reduced = at::div(reduced,ctx.world_size());
                }

                if (!this->output_save_name.empty()) {
                    this->add_matrix_state(output_save_name, reduced, ctx, true);
                }

                return reduced;
            }
        };
    }
}
