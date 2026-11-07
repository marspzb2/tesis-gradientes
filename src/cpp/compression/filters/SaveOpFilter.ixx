//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <memory>
#include <torch/torch.h>

export module SaveMatrixFilter;

import CompressionFilter;

using namespace compression::base;
using torch::Tensor;

export namespace compression::filters {
    template<typename ValueT, typename EncMetaT>
    class SaveMatrixFilter : public CompressionFilter<ValueT, ValueT, NoValue, EncMetaT> {
    private:
        std::string state_name;

    public:
        explicit SaveMatrixFilter(const std::string &state_name)
            : state_name(state_name) {
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<NoValue> &metadata, NoValue *state,
                                  CompressionFilterCtx &ctx) override {
            ctx.put_state(this->state_name, std::make_unique<SavedMatrixState>(state_name, tensor));
            return tensor;
        }
    };

    template<typename ValueT, typename MetaEncT>
    inline constexpr auto f_save_matrix = FilterCtor<SaveMatrixFilter<ValueT, MetaEncT> >{};
}
