module;
#include <torch/torch.h>

export module BaseMethod;
import CompressionFilter;
using torch::Tensor;

export namespace compression::methods {
    class BaseCompressionMethod {
    public:
        virtual ~BaseCompressionMethod() {
        }

        virtual Tensor doMethod(Tensor &input, base::CompressionFilterCtx &filter_ctx) = 0;
    };
}
