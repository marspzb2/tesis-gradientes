add_library(utils)
target_compile_features(utils PUBLIC cxx_std_20)

target_sources(utils
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        bit_utils.ixx
        elias.ixx
        torch_scalar_utils.ixx
        BitConcatUtils.ixx
        BitSelectionUtils.ixx
        CommonConcepts.ixx
        CommonValidations.ixx
)
find_package(Torch REQUIRED CONFIG)
target_link_libraries(utils PRIVATE
        torch
        OpenMP::OpenMP_CXX
)
