# Has common serialization methods for Tensors
add_library(serialization)
target_compile_features(serialization PUBLIC cxx_std_20)

file(GLOB SERIALIZATION CONFIGURE_DEPENDS "*.ixx")


target_sources(serialization
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${SERIALIZATION}

)
find_package(Torch REQUIRED CONFIG)
target_link_libraries(serialization PRIVATE
        torch
        OpenMP::OpenMP_CXX
        utils
)
