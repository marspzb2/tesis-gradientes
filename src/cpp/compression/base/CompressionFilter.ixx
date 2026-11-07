//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
#include <spdlog/spdlog.h>
#include <functional>
#include <fmt/core.h>

export module CompressionFilter;
import torch_scalar_utils;
import Network;
import CommonValidations;

using torch::Tensor;

export namespace compression::base {
    struct NoValue {
    };

    class CompressionFilterBaseState {
    public:
        virtual ~CompressionFilterBaseState() = default;
    };

    class SavedMatrixState : public CompressionFilterBaseState {
    private:
        std::string _name;
        Tensor _matrix;

    public:
        SavedMatrixState(const std::string &name, const Tensor &matrix)
            : _name(name),
              _matrix(matrix) {
        }

        Tensor matrix() {
            return _matrix;
        };

        std::string name() {
            return _name;
        }
    };

    class CompressionFilterCtx {
        std::reference_wrapper<Network> network_;
        std::unordered_map<std::string, std::unique_ptr<CompressionFilterBaseState> > states_;
        uint world_size_;

    public:
        CompressionFilterCtx(int world_size) : network_(Network::instance()),
                                               states_(
                                                   std::unordered_map<std::string, std::unique_ptr<
                                                       CompressionFilterBaseState> >()),
                                               world_size_(world_size) {
        }

        CompressionFilterCtx(const std::reference_wrapper<Network> network, int world_size)
            : network_(network),
              states_(std::unordered_map<std::string, std::unique_ptr<CompressionFilterBaseState> >()),
              world_size_(world_size) {
        }

        std::reference_wrapper<Network> network() {
            return network_;
        }

        uint world_size() {
            return this->world_size_;
        }

        template<typename StateT>
        StateT *get_state(const std::string &state_name) {
            spdlog::debug("Getting state for '{}'", state_name);
            return dynamic_cast<StateT *>(this->states_.at(state_name).get());
        }

        bool is_state_present(const std::string &state_name) {
            return this->states_.find(state_name) != this->states_.end();
        }

        template<typename StateT>
        StateT *put_state(const std::string &state_name, std::unique_ptr<StateT> state) {
            StateT *raw_ptr = state.get();
            spdlog::debug("Setting state for '{}'", state_name);

            auto base_ptr = reinterpret_cast<base::CompressionFilterBaseState *>(state.release());
            this->states_[state_name].reset(base_ptr);

            return raw_ptr;
        }
    };

    /**
     * La clase base de todos los filtros de compresion y decompression.
     * @tparam InputT La entrada del filtro.
     * @tparam OutputT  La salida del filtro.
     * @tparam FilterMetaT La metadata del filtro de forma de clase.
     * @tparam InputEncMetaT Como se codifica la entrada de la metadata.
     * @tparam OutputEncMetaT Como se codifica la salida de la metadata.
     * @tparam StateT El valor del estado del Filtro.
     */
    template<typename InputT,
        typename OutputT=InputT,
        typename FilterMetaT = NoValue,
        typename InputEncMetaT = int64_t,
        typename OutputEncMetaT = InputEncMetaT,
        typename StateT = NoValue>
    class CompressionFilter {
    public:
        using InputType = InputT;
        using OutputType = OutputT;
        using MetaType = FilterMetaT;
        using EncodedInpMetaType = InputEncMetaT;
        using EncodedOutMetaType = OutputEncMetaT;
        using StateType = StateT;

        virtual ~CompressionFilter() = default;

        virtual Tensor internal_do_filter(const Tensor &tensor,
                                          const std::optional<FilterMetaT> &metadata,
                                          StateT *state,
                                          CompressionFilterCtx &ctx) {
            return tensor;
        };

        virtual Tensor internal_metadata(const Tensor &tensor,
                                         const Tensor &meta_tensor,
                                         const std::optional<FilterMetaT> &metadata,
                                         StateT *state,
                                         CompressionFilterCtx &ctx) {
            return meta_tensor;
        }

        virtual bool has_state(const Tensor &tensor, const Tensor &previousMetadata,
                               const std::optional<FilterMetaT> &metadata) {
            return false;
        }

        bool refresh_state(const Tensor &tensor, const Tensor &previousMetadata,
                           CompressionFilterCtx &compression_ctx) {
            return false;
        }

        virtual std::string state_name(const Tensor &tensor) {
            return std::string{};
        }

        virtual std::unique_ptr<StateT> create_state(const Tensor &tensor, const Tensor &previousMetadata) {
            return nullptr;
        }

        SavedMatrixState *add_matrix_state(std::string name,
                                           Tensor matrix,
                                           CompressionFilterCtx &compression_ctx,
                                           bool rewrite = true) {
            if (rewrite || !compression_ctx.is_state_present(name)) {
                return compression_ctx.put_state(name, std::make_unique<SavedMatrixState>(name, matrix));
            }
            return compression_ctx.get_state<SavedMatrixState>(name);
        }


        StateT *get_or_create_state(const Tensor &tensor,
                                    const Tensor &previousMetadata,
                                    CompressionFilterCtx &compression_ctx,
                                    bool always_recreate = false) {
            StateT *state = nullptr;
            if (!always_recreate && compression_ctx.is_state_present(this->state_name(tensor))) {
                state = compression_ctx.get_state<StateT>(this->state_name(tensor));
            } else {
                state = compression_ctx.put_state<StateT>(this->state_name(tensor),
                                                          this->create_state(tensor, previousMetadata));
            }
            return state;
        }

        virtual std::optional<FilterMetaT> parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) {
            return std::nullopt;
        }

        virtual bool recreate_state(const Tensor &tensor, const Tensor &meta_tensor,
                                    const CompressionFilterCtx &compression_ctx) {
            return false;
        }

        virtual std::string filter_init_debug_info(const Tensor &tensor,
                                                   const Tensor &previousMetadata,
                                                   const CompressionFilterCtx &compression_ctx) {
            return fmt::format("{}", typeid(*this).name());
        }

        virtual std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                                     CompressionFilterCtx &compression_ctx) {
            spdlog::debug("Ejecutando filtro '{}'",
                          this->filter_init_debug_info(tensor, meta_tensor, compression_ctx));
            assert(tensor.dtype() == compression::utils::get_torch_dtype());

            std::optional<FilterMetaT> metadata = parse_metadata(tensor, meta_tensor);

            StateT *state = nullptr;
            if (this->has_state(tensor, meta_tensor, metadata)) {
                state = get_or_create_state(tensor, meta_tensor, compression_ctx,
                                            this->recreate_state(tensor, meta_tensor, compression_ctx));
            }
            auto result = std::make_tuple(this->internal_do_filter(tensor, metadata, state, compression_ctx),
                                          this->internal_metadata(tensor, meta_tensor, metadata, state,
                                                                  compression_ctx));
            spdlog::debug("Filtro finalizado '{}'",
                          this->filter_init_debug_info(tensor, meta_tensor, compression_ctx));
            return result;
        };
    };


    class ReductionStrategy {
    public:
        virtual ~ReductionStrategy() = default;

        virtual std::tuple<Tensor, Tensor> reduce(const std::vector<std::vector<Tensor> > &values,
                                                  CompressionFilterCtx &ctx) =0;
    };

    /**
     * Una estategia reducción que usa un filtro para sacar la salida.
     * @tparam InpValueT La entrada del filtro (lo que llega de los otros valores).
     * @tparam InpEncodingT La codificación del encoding del filtro.
     * @tparam OutputValueT La salida del filtro.
     * @tparam OutEncodingType La salida de la codificacion.
     * @tparam MetaT El tipo de la metadata.
     * @tparam StateT El estado del filtro.
     */
    template<typename FilterT>
        requires std::derived_from<FilterT,
            CompressionFilter<
                typename FilterT::InputType,
                typename FilterT::OutputType,
                typename FilterT::MetaType,
                typename FilterT::EncodedInpMetaType,
                typename FilterT::EncodedOutMetaType,
                typename FilterT::StateType> >
    class FilterReductionStrategy : public ReductionStrategy {
        using OutputValueT = typename FilterT::OutputType;
        using FilterResult = std::tuple<Tensor, Tensor>;
        using CombineFunction = std::function<FilterResult(const FilterResult &, const FilterResult &,
                                                           const std::vector<std::vector<Tensor> > &values)>;
        using EndTransformFn = std::function<FilterResult(const FilterResult &,
                                                          const std::vector<std::vector<Tensor> > &values)>;

    private:
        std::unique_ptr<FilterT> filter;
        CombineFunction combine;
        EndTransformFn end_transform;
        int ignore_filter_rank_;

    public:
        explicit FilterReductionStrategy(std::unique_ptr<FilterT> filter,
                                         CombineFunction combine,
                                         EndTransformFn end_transform,
                                         int ignore_filter_rank = -1
        )
            : filter(std::move(filter)), combine(combine), end_transform(end_transform),
              ignore_filter_rank_(ignore_filter_rank) {
        }

        std::tuple<Tensor, Tensor> reduce(const std::vector<std::vector<Tensor> > &values,
                                          CompressionFilterCtx &ctx) override {
            utils::validate_size_is(values, 2, "The vector must have at least two list");
            utils::validate_not_empty(values[0], "List values cannot be empty");
            utils::validate_same_size(values[0], values[1], "List mus share the smae size");

            bool first = true;
            FilterResult result;

            for (int rank = 0; rank < values.front().size(); rank++) {
                Tensor value_tensor = values[0][rank];
                Tensor value_metadata = values[1][rank];
                std::tuple current_value{value_tensor, value_metadata};
                if (ignore_filter_rank_ != rank) {
                    current_value = filter->do_filter(value_tensor, value_metadata, ctx);
                }
                if (first) {
                    result = current_value;
                    first = false;
                } else {
                    result = combine(result, current_value, values);
                }
            }
            return end_transform(result, values);
        };

        static std::unique_ptr<FilterReductionStrategy> meanReductionStrategy(
            std::unique_ptr<FilterT> filter, int ignore_filter_rank = -1) {
            return std::make_unique<FilterReductionStrategy<FilterT> >(
                std::move(filter),
                [](const FilterResult &a, const FilterResult &b,
                   const std::vector<std::vector<Tensor> > &values) {
                    return std::make_tuple(std::get<0>(a) + std::get<0>(b), std::get<1>(a));
                },
                [](const FilterResult &a, const std::vector<std::vector<Tensor> > &values) {
                    return std::make_tuple(std::get<0>(a) / (1. * values.front().size()), std::get<1>(a));
                },
                ignore_filter_rank);
        }
    };

    template<typename InputT, typename EncMetaT, typename OutputT=InputT>
    class SimpleFilterCls : public CompressionFilter<InputT, OutputT, NoValue, EncMetaT> {
        std::function<Tensor(const Tensor &, CompressionFilterCtx &)> filter_function;
        bool pass_result;
        std::optional<std::string> save_result_to;

    public:
        SimpleFilterCls(const std::function<Tensor(const Tensor &, CompressionFilterCtx &)> &filter_function,
                        bool pass_result, const std::optional<std::string> &save_result_to)
            : filter_function(filter_function),
              pass_result(pass_result),
              save_result_to(save_result_to) {
        }

        Tensor internal_do_filter(const Tensor &tensor,
                                  const std::optional<NoValue> &metadata,
                                  NoValue *state,
                                  CompressionFilterCtx &ctx) override {
            auto res = filter_function(tensor, ctx);
            if (save_result_to) {
                ctx.put_state(save_result_to.value(),
                              std::make_unique<SavedMatrixState>(save_result_to.value(),
                                                                 res));
            }
            if (pass_result) {
                return res;
            }
            return tensor;
        };
    };

    template<typename InputT, typename EncMetaT, typename OutputT=InputT>
    auto MapFilter(
        const std::function<Tensor(const Tensor &, CompressionFilterCtx &)> &filter_function,
        bool pass_result = false,
        std::optional<std::string> save_result_to = std::nullopt
    ) {
        return std::make_unique<SimpleFilterCls<InputT, EncMetaT, OutputT> >(
            filter_function, pass_result, save_result_to);
    }

    /**
     * Un truco, a raiz de la sobrecarga del operador permite crear funciones de orden superior sobre constructores,
     * para eliminar el std::make_unique de la definicion.
     * @tparam FilterT El tipo del filtro.
     */
    template<typename FilterT>
    struct FilterCtor {
        /**
         * @tparam Args los parametrso del constructor.
         */
        template<class... Args>
        auto operator()(Args &&... args) const {
            return std::make_unique<FilterT>(std::forward<Args>(args)...);
        }
    };
}
