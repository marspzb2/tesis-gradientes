//
// Created by Marcos Perez on 7/6/26.
//

//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <torch/torch.h>

#include <utility>
export module AllReduceFilter;

import CompressionFilter;
using namespace compression::base;
using torch::Tensor;
using torch::TensorOptions;

export
{
    namespace compression::filters {
        template<typename ValueT, typename EncodingT>
        class AllReduceFilter :public CompressionFilter<ValueT, ValueT, NoValue, EncodingT> {
        private:
            std::string output_save_name;

        public:
            explicit AllReduceFilter(const std::string &  output_save_name)
                : output_save_name(output_save_name) {
            }

            Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<NoValue> &metadata,
                                      NoValue *state, CompressionFilterCtx &ctx) override {

                assert(ctx.network() != nullptr);
                assert(ctx.network().get().get_process_group() != nullptr);

                auto work = ctx.network().get().all_reduce(tensor);
                work->wait();
                Tensor reduced = work->result()[0] / ctx.world_size();

                if (!this->output_save_name.empty()) {
                    this->add_matrix_state(output_save_name, tensor, ctx, true);
                }

                return reduced;
            }
        };
    }
}
