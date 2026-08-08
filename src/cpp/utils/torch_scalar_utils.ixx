//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
export module torch_scalar_utils;

export
{
    namespace compression::utils {
        template<typename T>
        struct torch_type_traits {
            static_assert(sizeof(T) == 0, "Tipo de dato no soportado por LibTorch.");
        };

        template<>
        struct torch_type_traits<uint64_t> {
            static constexpr torch::ScalarType value = torch::kUInt64;
        };

        template<>
        struct torch_type_traits<float> {
            static constexpr torch::ScalarType value = torch::kFloat32;
        };

        template<>
        struct torch_type_traits<double> {
            static constexpr torch::ScalarType value = torch::kFloat64;
        };

        template<>
        struct torch_type_traits<int> {
            static constexpr torch::ScalarType value = torch::kInt32;
        };

        template<>
        struct torch_type_traits<int64_t> {
            static constexpr torch::ScalarType value = torch::kInt64;
        };

        template<>
        struct torch_type_traits<bool> {
            static constexpr torch::ScalarType value = torch::kBool;
        };

        template<>
        struct torch_type_traits<unsigned char> {
            static constexpr torch::ScalarType value = torch::kUInt8;
        };

        // 3. La función wrapper que vas a usar en tu código
        template<typename T>
        [[nodiscard]] constexpr torch::ScalarType get_torch_dtype() noexcept {
            return torch_type_traits<T>::value;
        }
    }
}
