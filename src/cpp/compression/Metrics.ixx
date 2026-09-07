module;
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <torch/torch.h>

export module Metrics;

using torch::Tensor;

export namespace compression::metrics {
    struct VectorMetricOptions {
        std::string name;
        bool track_norm_0{true};
        bool track_norm_1{true};
        bool track_norm_2{true};
        bool track_norm_inf{true};
    };

    class VectorMetric {
    private:
        const VectorMetricOptions options_;
        mutable std::mutex mutex_;

        std::size_t registered_chunk_count_{0};
        double norm_0_sum_{0};
        double norm_1_sum_{0};
        double norm_2_squared_sum_{0};
        double norm_inf_max_{0};
        uint64_t bytes_{0};
        size_t total_elements_{0};

    public:
        explicit VectorMetric(VectorMetricOptions options = {})
            : options_(std::move(options)) {
        }

        void register_vector(const Tensor &tensor) {
            if (!tensor.defined()) {
                throw std::invalid_argument(
                    "VectorMetric::register_vector: undefined tensor");
            }

            // Las operaciones costosas se realizan fuera de la seccion critica.
            const Tensor values = tensor.detach().to(torch::kFloat64);
            const Tensor absolute_values = values.abs();
            const Tensor zero = torch::zeros({}, values.options());

            const Tensor statistics = torch::stack({
                options_.track_norm_0
                    ? values.ne(0).sum().to(torch::kFloat64)
                    : zero,
                options_.track_norm_1
                    ? absolute_values.sum()
                    : zero,
                options_.track_norm_2
                    ? values.square().sum()
                    : zero,
                options_.track_norm_inf && values.numel() != 0
                    ? absolute_values.max()
                    : zero
            });

            // Una sola sincronizacion GPU-CPU.
            const Tensor cpu_statistics =
                statistics.to(torch::kCPU).contiguous();
            const double *statistics_data =
                cpu_statistics.data_ptr<double>();

            // Solo la actualizacion de los acumuladores queda serializada.
            const std::scoped_lock lock(mutex_);
            norm_0_sum_ += statistics_data[0];
            norm_1_sum_ += statistics_data[1];
            norm_2_squared_sum_ += statistics_data[2];
            norm_inf_max_ =
                std::max(norm_inf_max_, statistics_data[3]);
            bytes_ += tensor.numel() * tensor.element_size();
            ++registered_chunk_count_;
            total_elements_ += tensor.numel();
        }

        void reset() {
            const std::scoped_lock lock(mutex_);
            registered_chunk_count_ = 0;
            norm_0_sum_ = 0;
            norm_1_sum_ = 0;
            norm_2_squared_sum_ = 0;
            norm_inf_max_ = 0;
            bytes_ = 0;
            total_elements_ = 0;
        }

        [[nodiscard]] const std::string &name() const noexcept {
            return options_.name;
        }

        [[nodiscard]] std::size_t registered_chunk_count() const {
            const std::scoped_lock lock(mutex_);
            return registered_chunk_count_;
        }

        [[nodiscard]] double norm_0() const {
            const std::scoped_lock lock(mutex_);
            return norm_0_sum_;
        }

        [[nodiscard]] double norm_1() const {
            const std::scoped_lock lock(mutex_);
            return norm_1_sum_;
        }

        [[nodiscard]] double norm_2() const {
            const std::scoped_lock lock(mutex_);
            return std::sqrt(norm_2_squared_sum_);
        }

        [[nodiscard]] double norm_inf() const {
            const std::scoped_lock lock(mutex_);
            return norm_inf_max_;
        }

        [[nodiscard]] uint64_t bytes() const {
            const std::scoped_lock lock(mutex_);
            return bytes_;
        }

        [[nodiscard]] size_t total_elements() const {
            const std::scoped_lock lock(mutex_);
            return total_elements_;
        }
    };

    struct NumberTrackingOptions {
        std::string name;
        double running_average_factor{1};
        bool reset_running_average_always{false};
    };

    template<typename NumberT>
        requires std::is_arithmetic_v<NumberT>
    class NumberTracking {
    private:
        const NumberTrackingOptions options_;
        mutable std::mutex mutex_;
        NumberT number_{0};
        uint64_t total_calls_{0};
        double running_average_{0};
        double aggregated_number_{0};
        NumberT max_ = std::numeric_limits<NumberT>::lowest();
        NumberT min_ = std::numeric_limits<NumberT>::max();

    public:
        explicit NumberTracking(NumberTrackingOptions options)
            : options_(std::move(options)) {
            if (options_.running_average_factor < 0 ||
                options_.running_average_factor > 1) {
                throw std::invalid_argument(
                    "NumberTracking: running_average_factor must be in [0, 1]");
            }
        }

        NumberTracking &register_number(NumberT number, bool register_aggregation = true,
                                        bool register_min_max = true) {
            const std::scoped_lock lock(mutex_);
            number_ = number;

            if (total_calls_ == 0) {
                running_average_ = static_cast<double>(number);
            } else {
                running_average_ =
                        running_average_ *
                        (1 - options_.running_average_factor) +
                        static_cast<double>(number) *
                        options_.running_average_factor;
            }
            if (register_aggregation) {
                ++total_calls_;
                aggregated_number_ += static_cast<double>(number);
            }
            if (register_min_max) {
                if (max_ < number) {
                    max_ = number;
                }
                if (min_ > number) {
                    min_ = number;
                }
            }
            return *this;
        }

        NumberTracking &register_min_max(NumberT min, NumberT max) {
            const std::scoped_lock lock(mutex_);
            min_ = std::min(min, this->min_);
            max_ = std::max(max, this->max_);
            return *this;
        }

        NumberTracking &add_aggregation(NumberT number, uint64_t total_calls) {
            const std::scoped_lock lock(mutex_);
            this->total_calls_ += total_calls;
            this->aggregated_number_ += static_cast<double>(number);
            return *this;
        }

        void reset(bool full_reset = false) {
            const std::scoped_lock lock(mutex_);
            number_ = NumberT{0};
            max_ = std::numeric_limits<NumberT>::lowest();
            min_ = std::numeric_limits<NumberT>::max();
            if (full_reset || options_.reset_running_average_always) {
                total_calls_ = 0;
                running_average_ = 0;
                aggregated_number_ = 0;
            }
        }

        [[nodiscard]] const std::string &name() const noexcept {
            return options_.name;
        }


        [[nodiscard]] NumberT value() const {
            const std::scoped_lock lock(mutex_);
            return number_;
        }

        [[nodiscard]] NumberT min() const {
            const std::scoped_lock lock(mutex_);
            return min_;
        }

        [[nodiscard]] NumberT max() const {
            const std::scoped_lock lock(mutex_);
            return max_;
        }

        [[nodiscard]] double running_average() const {
            const std::scoped_lock lock(mutex_);
            return running_average_;
        }

        [[nodiscard]] double aggregated_number() const {
            const std::scoped_lock lock(mutex_);
            return aggregated_number_;
        }

        [[nodiscard]] uint64_t total_calls() const {
            const std::scoped_lock lock(mutex_);
            return total_calls_;
        }

        [[nodiscard]] double avg() const {
            const std::scoped_lock lock(mutex_);
            return total_calls_ == 0
                       ? 0
                       : aggregated_number_ /
                         static_cast<double>(total_calls_);
        }
    };

    class Metrics {
    private:
        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::unique_ptr<VectorMetric> >
        vector_metrics_;
        std::unordered_map<
            std::string,
            std::unique_ptr<NumberTracking<double> > >
        number_metrics_;

        Metrics() = default;

    public:
        Metrics(const Metrics &) = delete;

        Metrics &operator=(const Metrics &) = delete;

        Metrics(Metrics &&) = delete;

        Metrics &operator=(Metrics &&) = delete;

        static Metrics &metrics() {
            static Metrics metrics;
            return metrics;
        }

        VectorMetric &vector(VectorMetricOptions options) {
            if (options.name.empty()) {
                throw std::invalid_argument(
                    "Metrics::vector_metric: name must not be empty");
            }

            const std::scoped_lock lock(mutex_);
            const std::string name = options.name;
            auto [position, inserted] = vector_metrics_.try_emplace(
                name,
                nullptr);

            if (inserted) {
                position->second =
                        std::make_unique<VectorMetric>(std::move(options));
            }

            return *position->second;
        }

        VectorMetric &vector(const std::string &name) {
            const std::scoped_lock lock(mutex_);
            const auto position = vector_metrics_.find(name);

            if (position == vector_metrics_.end()) {
                throw std::out_of_range(
                    "Metrics::vector: metric not found: " + name);
            }

            return *position->second;
        }

        NumberTracking<double> &number(NumberTrackingOptions options) {
            if (options.name.empty()) {
                throw std::invalid_argument(
                    "Metrics::number_metric: name must not be empty");
            }

            const std::scoped_lock lock(mutex_);
            const std::string name = options.name;
            auto [position, inserted] = number_metrics_.try_emplace(
                name,
                nullptr);

            if (inserted) {
                position->second =
                        std::make_unique<NumberTracking<double> >(
                            std::move(options));
            }

            return *position->second;
        }

        NumberTracking<double> &number(const std::string &name) {
            const std::scoped_lock lock(mutex_);
            const auto position = number_metrics_.find(name);

            if (position == number_metrics_.end()) {
                throw std::out_of_range(
                    "Metrics::number: metric not found: " + name);
            }

            return *position->second;
        }

        [[nodiscard]] bool has_vector_metric(std::string name) const {
            return vector_metrics_.contains(name);
        }

        [[nodiscard]] bool has_number_metric(std::string name) const {
            return number_metrics_.contains(name);
        }

        void reset(bool full_reset = false) {
            const std::scoped_lock lock(mutex_);

            for (auto &[name, metric]: vector_metrics_) {
                metric->reset();
            }
            for (auto &[name, metric]: number_metrics_) {
                metric->reset(full_reset);
            }
        }
    };
}
