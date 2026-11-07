//
// Created by Marcos Perez on 6/6/26.
//

module;
#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/cat.h>

export module ShapeFilters;
import CompressionFilter;
export namespace compression::filters {
    template<typename ValueT, typename EncMetadataT>
    class FlattenShapeForward : public base::CompressionFilter<ValueT, ValueT, std::vector<int64_t>, EncMetadataT> {
    public:
        at::Tensor internal_do_filter(const at::Tensor &tensor,
                                      const std::optional<std::vector<int64_t> > &metadata,
                                      base::NoValue *state,
                                      base::CompressionFilterCtx &ctx) override {
            return tensor.flatten();
        }

        at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                     const std::optional<std::vector<int64_t> > &metadata,
                                     base::NoValue *state,
                                     base::CompressionFilterCtx &ctx) override {
            int dimensions = metadata->size();
            at::Tensor values = at::zeros({dimensions + 1}, meta_tensor.options());
            values[0] = static_cast<EncMetadataT>(dimensions);
            for (int i = 0; i < metadata->size(); ++i) {
                values[i + 1] = static_cast<EncMetadataT>(metadata->at(i));
            }
            return at::cat({values, meta_tensor}, 0);
        }

        std::optional<std::vector<int64_t> > parse_metadata(const at::Tensor &tensor,
                                                            const at::Tensor &previousMetadata) override {
            return std::vector(tensor.sizes().begin(), tensor.sizes().end());
        }
    };

    template<typename ValueT, typename EncMetadataT>
    class FlattenShapeBack : public base::CompressionFilter<ValueT, ValueT, std::vector<int64_t>, EncMetadataT> {
    public:
        at::Tensor internal_do_filter(const at::Tensor &tensor,
                                      const std::optional<std::vector<int64_t> > &metadata,
                                      base::NoValue *state,
                                      base::CompressionFilterCtx &ctx) override {
            return tensor.reshape(metadata.value());
        }

        at::Tensor internal_metadata(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                     const std::optional<std::vector<int64_t> > &metadata,
                                     base::NoValue *state,
                                     base::CompressionFilterCtx &ctx) override {
            int64_t dimensions = metadata->size();

            return meta_tensor.slice(0, dimensions + 1);
        }

        std::optional<std::vector<int64_t> > parse_metadata(const at::Tensor &tensor,
                                                            const at::Tensor &previousMetadata) override {
            std::vector<int64_t> metadata{};
            int64_t dimensions = static_cast<int64_t>(previousMetadata[0].item<EncMetadataT>());

            for (int64_t i = 0; i < dimensions; ++i) {
                metadata.push_back(static_cast<int64_t>(previousMetadata[i + 1].item<EncMetadataT>()));
            }
            return metadata;
        }
    };

    /**
     * Aplana el tensor de datos, devolviendo el tensor aplanado y dejando en los metadatos el shape original.
     * @tparam ValueT El tipo de valor de los datos.
     * @tparam EncMetadataT  El tipo de valor de la metadata.
     */
    template<typename ValueT, typename EncMetadataT>
    inline constexpr auto f_flatten_tensor_forward = base::FilterCtor<FlattenShapeForward<ValueT, EncMetadataT> >{};

    /**
     * Desenvuelve el aplanado del tensor.
     * @tparam ValueT El tipo de valor de los datos.
     * @tparam EncMetadataT  El tipo de valor de la metadata.
     */
    template<typename ValueT, typename EncMetadataT>
    inline constexpr auto f_flatten_tensor_backward = base::FilterCtor<FlattenShapeForward<ValueT, EncMetadataT> >{};
}
