module;
#include <cstdlib>
#include <string_view>
#include <torch/torch.h>

export module ErrorFeedbackFilters;

import CompressionFilter;
using torch::Tensor;
using compression::base::CompressionFilter;
using compression::base::NoValue;
using compression::base::SavedMatrixState;
#define MATRIX_STATE_NAME "ERROR_FEEDBACK"
#define ORIGINAL_MATRIX_STATE_NAME "ERROR_FEEDBACK_ORIGINAL"
export namespace compression::filters {
    template<typename ValueT, typename EncMetaT>
    class ApplyErrorFeedback : public CompressionFilter<ValueT, ValueT, NoValue, EncMetaT> {
    private:
        std::string param_name;
        bool is_enabled;

        static bool error_feedback_enabled() {
            const char *value = std::getenv("ERROR_FEEDBACK_ENABLED");
            return value != nullptr && std::string_view(value) == "true";
        }

    public:
        explicit ApplyErrorFeedback(const std::string &param_name)
            : param_name(param_name),
              is_enabled(error_feedback_enabled()) {
        }

        ApplyErrorFeedback()
            : param_name("INPUT"),
              is_enabled(error_feedback_enabled()) {
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<NoValue> &metadata, NoValue *state,
                                  base::CompressionFilterCtx &ctx) {
            Tensor apply_tensor;
            if (param_name == "INPUT" || param_name == "") {
                apply_tensor = tensor;
            } else {
                assert(ctx.is_state_present(param_name));
                apply_tensor = ctx.get_state<SavedMatrixState>(param_name)->matrix();
            }

            if (is_enabled && ctx.is_state_present(MATRIX_STATE_NAME)) {
                base::SavedMatrixState *errorData = ctx.get_state<base::SavedMatrixState>(MATRIX_STATE_NAME);
                auto resultTensor = apply_tensor + errorData->matrix();
                ctx.put_state(
                    ORIGINAL_MATRIX_STATE_NAME,
                    std::make_unique<SavedMatrixState>(ORIGINAL_MATRIX_STATE_NAME, resultTensor));
                return resultTensor;
            } else {
                ctx.put_state(
                    ORIGINAL_MATRIX_STATE_NAME,
                    std::make_unique<SavedMatrixState>(ORIGINAL_MATRIX_STATE_NAME, apply_tensor));
                return apply_tensor;
            }
        }
    };

    template<typename ValueT, typename EncMetaT>
    class SaveErrorFeedback : public CompressionFilter<ValueT, ValueT, NoValue, EncMetaT> {
    private:
        std::function<Tensor(const Tensor &, const Tensor &, base::CompressionFilterCtx &)> decompress;

    public:
        explicit SaveErrorFeedback(
            const std::function<Tensor(const Tensor &, const Tensor &, base::CompressionFilterCtx &)> &decompress)
            : decompress(decompress) {
        }

        std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                             base::CompressionFilterCtx &compression_ctx) {
            SavedMatrixState *original_matrix = compression_ctx.get_state<SavedMatrixState>(ORIGINAL_MATRIX_STATE_NAME);
            if (original_matrix != nullptr) {
                Tensor localRepresentation = decompress(tensor, meta_tensor, compression_ctx);
                compression_ctx.put_state(MATRIX_STATE_NAME,
                                          std::make_unique<SavedMatrixState>(
                                              MATRIX_STATE_NAME, original_matrix->matrix() - localRepresentation));
            }

            return std::make_tuple(tensor, meta_tensor);
        }
    };
}
