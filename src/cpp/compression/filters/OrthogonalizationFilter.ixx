//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <torch/torch.h>
export module OrthogonalizationFilter;
import CompressionFilter;
using namespace compression::base;
using torch::Tensor;
using torch::TensorOptions;

export
{
    namespace compression::filters {
        template<typename ValueT, typename EncodingT>
        class OrthogonalizationFilter : public CompressionFilter<
                    ValueT, ValueT, NoValue, EncodingT> {
        private:
            std::string matrix_name{};

        public:
            explicit OrthogonalizationFilter(const std::string &matrix_name)
                : matrix_name(matrix_name) {
            }

            at::Tensor internal_do_filter(const at::Tensor &tensor, const std::optional<NoValue> &metadata,
                                          NoValue *state, CompressionFilterCtx &ctx) override {
                auto [ortho,r] = torch::linalg_qr(tensor, "reduced");
                //Tensor ortho = tensor;

                if (!this->matrix_name.empty()) {
                    this->add_matrix_state(matrix_name,ortho,ctx,true);
                }
                return ortho;
            }
        };
    }
}
