//
// Created by Marcos Perez on 7/6/26.
//
module;
#include <torch/torch.h>

export module MatrixMultiplyFilter;

import CompressionFilter;
import MatrixInitFilter;

using namespace compression::base;
using torch::Tensor;

export namespace compression::filters {
    enum class MatrixTransformType {
        NONE,
        TRANSPOSE,
        INVERSE,
        RESHAPE
    };

    enum class MatrixBinaryOpType {
        DOT,
        ADD,
        SUB,
        H_PROD,
        H_DIV
    };

    template<typename ValueT, typename EncMetaT>
    class MatOperationFilter : public CompressionFilter<ValueT, ValueT, NoValue, EncMetaT> {
    private:
        std::string left_name;
        std::string right_name;
        std::string result_name;
        MatrixTransformType left_transform;
        MatrixTransformType right_transform;
        MatrixBinaryOpType operation;
        bool pass_;

    public:
        virtual ~MatOperationFilter() = default;

        MatOperationFilter(const std::string &left_name, const std::string &right_name,
                           const std::string &result_name,
                           MatrixBinaryOpType operation,
                           MatrixTransformType left_transform = MatrixTransformType::NONE,
                           MatrixTransformType right_transform = MatrixTransformType::NONE,
                           bool pass = true)
            : left_name(left_name),
              right_name(right_name),
              result_name(result_name),
              left_transform(left_transform),
              right_transform(right_transform),
              operation(operation),
              pass_(pass) {
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<NoValue> &metadata, NoValue *state,
                                  CompressionFilterCtx &ctx) override {
            Tensor a = get_tensor(left_name, tensor, left_transform, ctx, tensor);
            Tensor b = get_tensor(right_name, tensor, right_transform, ctx, a);

            Tensor result = a;
            switch (operation) {
                case MatrixBinaryOpType::ADD:
                    result = a + b;
                    break;
                case MatrixBinaryOpType::SUB:
                    result = a - b;
                    break;
                case MatrixBinaryOpType::H_PROD:
                    result = a * b;
                    break;
                case MatrixBinaryOpType::H_DIV:
                    result = a / b;
                    break;
                case MatrixBinaryOpType::DOT:
                    result = a.matmul(b);
                    break;
                default:
                    throw std::logic_error("Unsupported operation");
            }
            if (!result_name.empty()) {
                this->add_matrix_state(result_name, result, ctx);
            }
            if (pass_) {
                return result;
            }
            return tensor;
        }

        Tensor get_tensor(const std::string &name, const Tensor &inputTensor, MatrixTransformType transform,
                          CompressionFilterCtx &ctx,
                          Tensor prev) {
            Tensor t = inputTensor;
            if (name.empty() || name == "" || name == "INPUT") {
                t = inputTensor;
            } else if (name == "IDENTITY") {
                t = torch::eye(prev.sizes()[0]);
            } else {
                t = ctx.get_state<SavedMatrixState>(name)->matrix();
            }
            switch (transform) {
                case MatrixTransformType::TRANSPOSE:
                    t = t.t();
                    break;
                case MatrixTransformType::INVERSE:
                    t = t.inverse();
                    break;
                case MatrixTransformType::NONE:
                default:
                    break;
            }

            return t;
        }

        static auto mat_dot(const std::string &left_name, const std::string &right_name,
                            const std::string &result_name,
                            MatrixTransformType left_transform = MatrixTransformType::NONE,
                            MatrixTransformType right_transform = MatrixTransformType::NONE,
                            bool pass = true
        ) {
            return std::make_unique<MatOperationFilter>(left_name, right_name, result_name, MatrixBinaryOpType::DOT,
                                                        left_transform,
                                                        right_transform, pass);
        }

        static auto mat_had(const std::string &left_name, const std::string &right_name,
                            const std::string &result_name,
                            MatrixTransformType left_transform = MatrixTransformType::NONE,
                            MatrixTransformType right_transform = MatrixTransformType::NONE
        ) {
            return std::make_unique<MatOperationFilter>(left_name, right_name, result_name,
                                                        MatrixBinaryOpType::H_PROD,
                                                        left_transform,
                                                        right_transform);
        }

        static auto mat_add(const std::string &left_name, const std::string &right_name,
                            const std::string &result_name,
                            MatrixTransformType left_transform = MatrixTransformType::NONE,
                            MatrixTransformType right_transform = MatrixTransformType::NONE
        ) {
            return std::make_unique<MatOperationFilter>(left_name, right_name, result_name, MatrixBinaryOpType::ADD,
                                                        left_transform,
                                                        right_transform);
        }

        static auto mat_sub(const std::string &left_name, const std::string &right_name,
                            const std::string &result_name,
                            MatrixTransformType left_transform = MatrixTransformType::NONE,
                            MatrixTransformType right_transform = MatrixTransformType::NONE
        ) {
            return std::make_unique<MatOperationFilter>(left_name, right_name, result_name, MatrixBinaryOpType::SUB,
                                                        left_transform,
                                                        right_transform);
        }
    };
}
