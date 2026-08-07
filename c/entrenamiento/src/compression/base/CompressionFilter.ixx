//
// Created by Marcos Perez on 6/6/26.
//
module;
#include <torch/torch.h>
export module CompressionFilter;
using torch::Tensor;
import torch_scalar_utils;
import Network;
export
{
    namespace compression::pipes {
        struct NoValue {
        };

        class CompressionPipeState {
        };

        class CompressionPipeCtx {
            std::shared_ptr<Network> network_;
            std::unordered_map<std::string, std::unique_ptr<CompressionPipeState> > states_;

        public:
            CompressionPipeCtx(const std::shared_ptr<Network> &network)
                : network_(network),
                  states_(std::unordered_map<std::string, std::unique_ptr<CompressionPipeState> >()) {
            }

            std::shared_ptr<Network> network() {
                return network_;
            }

            template<typename StateT>
            StateT *get_state(const std::string &state_name) {
                return dynamic_cast<StateT *>(this->states_.at(state_name).get());
            }

            bool is_state_present(const std::string &state_name) {
                return this->states_.find(state_name) != this->states_.end();
            }

            template<typename StateT>
            std::unique_ptr<StateT> put_state(const std::string &state_name, std::unique_ptr<StateT> state) {
                return this->states_[state_name] = std::move(state);
            }
        };

        template<typename InputT,
            typename OutputT=InputT,
            typename FilterMetaT = NoValue,
            typename InputEncMetaT = int64_t,
            typename OutputEncMetaT = InputEncMetaT,
            typename StateT = NoValue>
        class CompressionFilter {
        public:
            virtual ~CompressionFilter() = default;

            virtual Tensor internal_do_filter(const Tensor &tensor,
                                              const std::optional<FilterMetaT> &metadata,
                                              StateT *state,
                                              CompressionPipeCtx &ctx) =0;

            virtual Tensor internal_metadata(const Tensor &tensor,
                                             const Tensor &meta_tensor,
                                             const std::optional<FilterMetaT> &metadata,
                                             StateT *state,
                                             CompressionPipeCtx &ctx) =0;

            virtual std::optional<FilterMetaT> get_metadata(const Tensor &tensor) {
                return std::nullopt;
            }

            bool has_state(const Tensor &tensor) {
                return false;
            }

            std::string state_name(const Tensor &tensor) {
                return std::string{};
            }

            virtual StateT create_state(const Tensor &tensor, const Tensor &previousMetadata) {
                return CompressionPipeState{};
            }

            StateT *get_or_create_state(const Tensor &tensor,
                                        const Tensor &previousMetadata,
                                        CompressionPipeCtx &compression_ctx) {
                StateT *state = nullptr;
                if (compression_ctx.is_state_present(this->state_name())) {
                    state = compression_ctx.get_state<StateT>(this->state_name());
                } else {
                    state = compression_ctx.put_state<StateT>(this->state_name(),
                                                              this->create_state(tensor, previousMetadata));
                }
                return state;
            }

            virtual std::optional<FilterMetaT> parse_metadata(const Tensor &tensor, const Tensor &previousMetadata) {
                return std::nullopt;
            }

            virtual std::tuple<Tensor, Tensor> do_filter(const Tensor &tensor, const Tensor &meta_tensor,
                                                         CompressionPipeCtx &compression_ctx) {
                assert(tensor.dtype() == compression::utils::get_torch_dtype());
                StateT *state = nullptr;
                if (this->has_state(tensor)) {
                    state = get_or_create_state(tensor, meta_tensor, compression_ctx);
                }
                std::optional<FilterMetaT> metadata = parse_metadata(tensor, meta_tensor);
                return std::make_tuple(this->internal_do_filter(tensor, metadata, state, compression_ctx),
                                       this->internal_metadata(tensor, meta_tensor, metadata, state, compression_ctx));
            };
        };
    }
}
