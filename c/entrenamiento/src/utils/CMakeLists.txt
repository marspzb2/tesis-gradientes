set(CMAKE_CXX_SCAN_FOR_MODULES ON)

add_library(compression-base)

target_sources(compression-base
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        BaseCompressorState.ixx
        ComponentCompressor.ixx
        QSGDCompressor.ixx
        compression.ixx

)
find_package(Torch REQUIRED CONFIG)
target_link_libraries(compression-base PRIVATE torch)
