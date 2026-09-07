module;
#include <vector>
#include <ATen/core/Tensor.h>
#include <stdexcept>
#include <concepts>

export module CommonValidations;
export namespace compression::utils {
    template<typename t, typename u>
    void validate_same_size(const std::vector<t> &a, const std::vector<u> &b,
                            std::string error_msg) {
        if (a.size() != b.size()) {
            throw std::invalid_argument(error_msg);
        }
    }


    template<typename T>
    void validate_same_size(const std::vector<T> &a, const at::Tensor &b,
                            std::string error_msg) {
        if (a.size() != b.numel()) {
            throw std::invalid_argument(error_msg);
        }
    }

    template<typename T>
    void validate_size_is(const std::vector<T> &a, int expected_size,
                          std::string error_msg) {
        if (a.size() != expected_size) {
            throw std::invalid_argument(error_msg);
        }
    }

    template<typename T>
    void validate_not_empty(const std::vector<T> &a,
                            std::string error_msg) {
        if (a.empty()) {
            throw std::invalid_argument(error_msg);
        }
    }

    void validate_flat_tensor(const at::Tensor &tensor, std::string error_msg) {
        if (tensor.dim() != 1) throw std::invalid_argument(error_msg);
    }

    template<std::integral t>
    void validate_gte(const t &value, const t &min, const std::string &msg) {
        if (value < min) throw std::invalid_argument(msg);
    }

    template<std::integral t>
    void validate_gt(const t &value, const t &min, const std::string &msg) {
        if (value <= min) throw std::invalid_argument(msg);
    }

    template<std::integral T>
    void validate_positive(const T &value, const std::string &msg) {
        validate_gt(value, T{0}, msg);
    }

    template<std::integral T>
    void validate_positive_or_zero(const T &value, const std::string &msg) {
        validate_gte<T>(value, T{0}, msg);
    }


    template<std::integral T>
    void validate_range(const T &value, const T &min, const T &max, const std::string &msg) {
        if (value < min || value > max) {
            throw std::invalid_argument(msg);
        }
    }
}
