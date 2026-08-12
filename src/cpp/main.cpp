#include <torch/torch.h>
#include <iostream>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
import EliasCoding;
import ChainFilter;
import all_filters;
import CompressionFilter;
using namespace compression::filters;

using namespace compression::base;

int main() {
    // Core check
    torch::Tensor tensor = torch::rand({2, 3}, torch::kFloat64);
    tensor = tensor * -1;
    std::cout << "Random Tensor:\n" << tensor << std::endl;
    //auto compressor = QSGDCompressor<double>(4);
    CompressionFilterCtx ctx(2);
    ctx.put_state("ORIGINAL", std::make_unique<SavedMatrixState>("ORIGINAL", tensor));
    double lattice_size = 0.01;

    auto qsgd_decoder =
            std::make_unique<EliasDecodingBackwardFilter<int64_t, int64_t, int64_t> >()
            >> std::make_unique<FixedQuantizationBackwardFilter<double, int64_t, int64_t> >(
                std::make_unique<RandomDitheringQuantization<double, int64_t> >(
                    std::make_unique<RegularQuantization<double, int64_t> >(lattice_size, true)))
            >> std::make_unique<NormalizationBackwardFilter<double, int64_t> >()
            >> std::make_unique<FlattenShapeBack<double, int64_t> >();
    //auto pepe= new FilterReductionStrategy(std::move(qsgd_decoder));
    auto qsgd_compresion = std::make_unique<FlattenShapeForward<double, int64_t> >()
                           >> std::make_unique<NormalizationForwardFilter<double, int64_t> >()
                           >> std::make_unique<FixedQuantizationForwardFilter<double, int64_t, int64_t> >(
                               std::make_unique<RandomDitheringQuantization<double, int64_t> >(
                                   std::make_unique<RegularQuantization<double, int64_t> >(lattice_size, true)))
                           >> std::make_unique<EliasEncodingForwardFilter<int64_t, int64_t, int64_t> >()
                           >> std::make_unique<AllGatherFilter<int64_t, int64_t> >("Lala",
                               FilterReductionStrategy<
                                   decltype(qsgd_decoder)::element_type
                               >::meanReductionStrategy(std::move(qsgd_decoder)));


    //For each parameter’s gradient M ∈ R
    // {n×m}, the aim of rank-r matrix approximation is to find matrices
    // P ∈ R {n×r}
    //and Q ∈ R{mxr}
    int r = 1;


    ///auto state = QSGDState();
    //std::cout << compressor.compress(tensor, state) << std::endl;

    //  auto qsgd = std::make_unique<ShapePipe<double, double, int64_t> >()
    //            >> std::make_unique<NormalizationPipe<double, double, int64_t> >()
    //          >> std::make_unique<FixedQuantizationPipe<double, int64_t, int64_t> >(1000, true)
    //        >> std::make_unique<EliasCodingPipe<int64_t, uint64_t, int64_t> >(torch::kInt64)
    //     '\ >> std::make_unique<TransferPipe<int64_t> >();

    //auto powerSgd =
    //       std::make_unique<MatrixMultiplyPipe>("Q",SND,{r})//Q ->P
    //      >>std::make_unique<ReducePipe>("P")//reduce P
    //     >>std::make_unique<Orthogonalize>()// Create P^
    //    >>std::make_unique<MatrixMultiplyPipe>("Q",FST)//new Q
    //    >>std::make_unique<ReducePipe>("Q");// \hat{Q}


    //std::make_unique<SharedCompressionMatrix()>
    //P ← MQ
    //4: P ← ALL REDUCE MEAN(P) . Now, P = W (M1 + . . . + MW )Q
    // 5: Pˆ ← ORTHOGONALIZE(P) . Orthonormal columns
    // >> std::make_unique<ORTHOGONALIZEPipe>()
    //>> (std::make_unique<MultipliyByOriginalTensor>()
    //   << std::make_unique<ORTHOGONALIZEPipe>())

    //  >> std::make_unique<ReducePipe<int64_t> >();
    //6: Q ← M>Pˆ
    //7: Q ← ALL REDUCE MEAN(Q)
    //auto powersgd =
    auto metadata_inicial = torch::empty({0}, torch::kInt64);
    //
    //QSGD compression
    //    auto [compressed,metadata] = qsgd_compresion->do_filter(tensor, metadata_inicial, ctx);
    //   std::cout << "End compression: \n" << compressed << "\n\n" << metadata << std::endl;
    // auto [decompressed,dec_meta] = qsgd_decoder->do_filter(compressed, metadata, ctx);
    // std::cout << "End decompression: \n" << decompressed << "\n\n" << metadata << std::endl;

    std::cout << "QSGD compression: " <<
            std::get<0>(qsgd_compresion->do_filter(tensor, metadata_inicial, ctx)) << std::endl;


    //std::cout << "Power sgd compression: " <<
    //        std::get<0>(power_sgd->do_filter(tensor, metadata_inicial, ctx)) << std::endl;

    //std::cout<<std::bit_cast<double>(compressor.compress(tensor,QSGDState())[0].item<int64_t>())<<std::endl;
    //std::cout<<std::get<0>(compression::utils::elias_omega_code<uint64_t>(10000))<<std::endl;
    // Multiplatform hardware acceleration check
    // std::cout << "Decompressed: " << compressor.decompress(compressor.compress(tensor, state), state) << std::endl;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA (NVIDIA GPU) is available." << std::endl;
    } else if (torch::mps::is_available()) {
        std::cout << "MPS (Apple Silicon GPU) is available." << std::endl;
    } else {
        std::cout << "Running on CPU execution backend." << std::endl;
    }
    return 0;
}
