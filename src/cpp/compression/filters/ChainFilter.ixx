//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
export module ChainFilter;
import CompressionFilter;
using namespace compression::base;

export
{
    namespace compression::filters {
        template<typename InputT,
            typename IntermediateT,
            typename OutputT,
            typename NextMetaT,
            typename PreviousMetaT,
            typename IncomingMetaT,
            typename IntermediateMetaT,
            typename OutMetaT,
        typename InpStateT,
        typename OutStateT>
        class ChainFilter : public CompressionFilter<InputT, OutputT, NoValue, IncomingMetaT, OutMetaT,NoValue> {
            std::unique_ptr<CompressionFilter<InputT, IntermediateT, PreviousMetaT, IncomingMetaT, IntermediateMetaT,InpStateT> >
            previous;
            std::unique_ptr<CompressionFilter<IntermediateT, OutputT, NextMetaT, IntermediateMetaT, OutMetaT,OutStateT> > next;

        public:
            ChainFilter(
                std::unique_ptr<CompressionFilter<InputT, IntermediateT, PreviousMetaT, IncomingMetaT,
                    IntermediateMetaT,InpStateT> > prev,
                std::unique_ptr<CompressionFilter<IntermediateT, OutputT, NextMetaT, IntermediateMetaT,
                    OutMetaT,OutStateT> > n)
                : previous(std::move(prev)),
                  next(std::move(n)) {
            }

            std::tuple<at::Tensor, at::Tensor> do_filter(const at::Tensor &tensor, const at::Tensor &meta_tensor,
                                                         CompressionFilterCtx &compression_ctx) override {
                auto [previous_result,previous_metadata] = previous->do_filter(tensor, meta_tensor, compression_ctx);
                return next->do_filter(previous_result, previous_metadata, compression_ctx);
            }
        };
    }

    /*
    * typename OutputT=InputT,
    typename FilterMetaT = NoValue,
    typename InputEncMetaT = int64_t,
    typename OutputEncMetaT = InputEncMetaT,
    typename StateT = NoValue*/
    namespace compression::base {
        // 1. El extractor de tipos universal basado en herencia
        // template<typename T>
        //struct filter_traits {
        //    static constexpr bool is_filter = false;
        //};

        // Buscamos coincidencia directamente con la clase base legítima CompressionFilter
        template<typename InT, typename OutT, typename FilterMeta, typename InEncMeta, typename OutEncMeta, typename
            StT>
        struct filter_traits_extractor {
            static constexpr bool is_filter = true;
            using InputT = InT;
            using OutputT = OutT;
            using FilterMetaT = FilterMeta;
            using InputEncMetaT = InEncMeta;
            using OutputEncMetaT = OutEncMeta;
            using StateT = StT;
        };

        // Función de ayuda para deducir la base CompressionFilter que Clang puede resolver
        template<typename InT, typename OutT, typename FilterMeta, typename InEncMeta, typename OutEncMeta, typename
            StT>
        filter_traits_extractor<InT, OutT, FilterMeta, InEncMeta, OutEncMeta, StT>
        deduce_filter_base(const CompressionFilter<InT, OutT, FilterMeta, InEncMeta, OutEncMeta, StT> *);

        // Caso base si no hereda de CompressionFilter
        void deduce_filter_base(...);

        // La magia para Clang: Evaluamos el tipo de retorno al pasarle un puntero de la clase derivada
        template<typename T>
        struct filter_traits {
            using deduced_t = decltype(deduce_filter_base(std::declval<T *>()));
            static constexpr bool is_filter = deduced_t::is_filter;

            // Si es un filtro válido, exponemos los tipos de forma pública y limpia
            using InputT = typename deduced_t::InputT;
            using OutputT = typename deduced_t::OutputT;
            using FilterMetaT = typename deduced_t::FilterMetaT;
            using InputEncMetaT = typename deduced_t::InputEncMetaT;
            using OutputEncMetaT = typename deduced_t::OutputEncMetaT;
            using StateT = typename deduced_t::StateT;
        };


        // 2. Tu operador >> limpio y 100% compatible con tus clases derivadas
        template<typename PipeLHS, typename PipeRHS>
            requires filter_traits<PipeLHS>::is_filter
                     && filter_traits<PipeRHS>::is_filter
                     && std::same_as<typename filter_traits<PipeLHS>::OutputT, typename filter_traits<PipeRHS>::InputT>
                     && std::same_as<typename filter_traits<PipeLHS>::OutputEncMetaT, typename filter_traits<
                         PipeRHS>::InputEncMetaT>
        auto operator >>(std::unique_ptr<PipeLHS> &&lhs, std::unique_ptr<PipeRHS> &&rhs) {
            using LTraits = filter_traits<PipeLHS>;
            using RTraits = filter_traits<PipeRHS>;

            using In = typename LTraits::InputT;
            using Mid = typename LTraits::OutputT; // Corregido: OutputT para mantener consistencia
            using Out = typename RTraits::OutputT;
            using PM = typename LTraits::FilterMetaT;
            using NM = typename RTraits::FilterMetaT;
            using InpEncMeta = typename LTraits::InputEncMetaT;
            using MidEncMeta = typename LTraits::OutputEncMetaT;
            using OutEncMeta = typename RTraits::OutputEncMetaT;
            using InpState =LTraits::StateT;
            using OutStateT= RTraits::StateT;

            return std::make_unique<filters::ChainFilter<In, Mid, Out, NM, PM, InpEncMeta, MidEncMeta, OutEncMeta,InpState,OutStateT> >(
                std::move(lhs),
                std::move(rhs)
            );
        }
    }
}
