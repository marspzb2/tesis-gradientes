module;
#include <concepts>
#include <concepts>
#include <ATen/core/Tensor.h>
export module CommonConcepts;

export namespace compression::utils {
    template<typename T>
    concept TensorT =
            std::same_as<std::remove_cvref_t<T>, at::Tensor>;
}
