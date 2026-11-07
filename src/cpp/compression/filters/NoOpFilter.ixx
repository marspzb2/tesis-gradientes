module;
#include <concepts>
#include <ATen/core/Tensor.h>


export module NoOpFilter;
import CompressionFilter;

using at::Tensor;
export namespace compression::filters {
    template<typename OriginalValueT, typename MetaT>
    class NoOpFilter
            : public base::CompressionFilter<OriginalValueT, OriginalValueT, base::NoValue, MetaT> {
    public:
        Tensor internal_do_filter(const Tensor &tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            return tensor;
        }
    };

    template<typename OriginalValueT, typename MetaT>
    inline constexpr auto f_pass_trough = base::FilterCtor<NoOpFilter<OriginalValueT, MetaT> >{};
}
