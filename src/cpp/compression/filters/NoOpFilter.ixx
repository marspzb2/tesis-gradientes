module;
#include <concepts>
#include <ATen/core/Tensor.h>

export module NoOpFilter;
import CompressionFilter;

using at::Tensor;
export namespace compression::filters {
    template<typename OriginalValueT>
    class NoOpFilter
            : public base::CompressionFilter<OriginalValueT> {
    public:
        Tensor internal_do_filter(const Tensor &tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            return tensor;
        }
    };
}
