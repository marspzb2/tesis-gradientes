module;
#include <torch/torch.h>

export module RegularLatticeQuantizationFilters;
import CompressionFilter;
import torch_scalar_utils;
using torch::Tensor;
export namespace compression::filters {
    template<typename OriginalValueT, typename QuantizationT>
    class ForwardRegularLatticeQuantizationFilter
            : public base::CompressionFilter<OriginalValueT, QuantizationT> {
    private:
        double lattice_size;
        bool dither;

    public:
        ForwardRegularLatticeQuantizationFilter(double lattice_size, bool dither)
            : lattice_size(lattice_size),
              dither(dither) {
            if (lattice_size <= 0) {
                throw std::invalid_argument("lattice size must be greater than 0");
            }
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            Tensor compressed_tensor = torch::floor(tensor / lattice_size);
            if (dither) {
                Tensor random = torch::rand_like(compressed_tensor);
                Tensor probability_of_dither = tensor / lattice_size - compressed_tensor;
                compressed_tensor = compressed_tensor + (random < probability_of_dither).to(compressed_tensor.dtype());
            }
            return compressed_tensor.to(utils::get_torch_dtype<QuantizationT>());
        }
    };


    template<typename OriginalValueT, typename QuantizationT>
    class BackwardRegularLatticeQuantizationFilter
            : public base::CompressionFilter<QuantizationT, OriginalValueT> {
    private:
        double lattice_size;

    public:
        BackwardRegularLatticeQuantizationFilter(double lattice_size)
            : lattice_size(lattice_size) {
            if (lattice_size <= 0) {
                throw std::invalid_argument("lattice size must be greater than 0");
            }
        }

        Tensor internal_do_filter(const Tensor &tensor, const std::optional<base::NoValue> &metadata,
                                  base::NoValue *state, base::CompressionFilterCtx &ctx) override {
            return tensor.to(utils::get_torch_dtype<OriginalValueT>()) * lattice_size;
        }
    };

    template<typename OriginalValueT, typename QuantizationT>
    std::unique_ptr<ForwardRegularLatticeQuantizationFilter<OriginalValueT, QuantizationT> >
    regular_lattice_quantization_filter(
        double lattice_size, bool dither) {
        return std::make_unique<ForwardRegularLatticeQuantizationFilter<OriginalValueT, QuantizationT> >(
            lattice_size, dither);
    }

    template<typename OriginalValueT, typename QuantizationT>
    std::unique_ptr<BackwardRegularLatticeQuantizationFilter<OriginalValueT, QuantizationT> >
    regular_lattice_dequantize_filter(
        double lattice_size) {
        return std::make_unique<BackwardRegularLatticeQuantizationFilter<OriginalValueT, QuantizationT> >(
            lattice_size);
    }
}
