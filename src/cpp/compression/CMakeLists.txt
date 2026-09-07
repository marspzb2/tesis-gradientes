find_package(Torch REQUIRED CONFIG)
find_package(pybind11 CONFIG REQUIRED)

find_library(
        TORCH_PYTHON_LIBRARY
        NAMES torch_python
        PATHS "${TORCH_INSTALL_PREFIX}/lib"
        NO_DEFAULT_PATH
        REQUIRED
)


#Base library

add_library(compression-base)
target_compile_features(compression-base PUBLIC cxx_std_20)
file(GLOB BASE "base/*.ixx")
file(GLOB FILTERS "filters/*.ixx")
target_sources(compression-base
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        Metrics.ixx
        Network.ixx
        ${BASE}
        ${FILTERS}
)

target_link_libraries(compression-base PRIVATE
        torch
        spdlog::spdlog
        utils
        serialization
)

target_link_libraries(compression-base PUBLIC
        OpenMP::OpenMP_CXX
)

#Compression methods
add_library(compression-methods)
target_compile_features(compression-methods PUBLIC cxx_std_20)
file(GLOB METHODS "methods/*.ixx")
target_sources(compression-methods
        PUBLIC
        FILE_SET CXX_MODULES
        FILES
        ${METHODS}
)

target_link_libraries(compression-methods PRIVATE
        torch
        spdlog::spdlog
        compression-base
)

#Export bindings

pybind11_add_module(compression_python
        methods/ExportMethods.cpp
)
set_target_properties(compression_python PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

set_property(TARGET compression_python PROPERTY CXX_SCAN_FOR_MODULES ON)
target_link_libraries(compression_python
        PUBLIC
        torch
        compression-base
        compression-methods
        Python::Python
        "${TORCH_PYTHON_LIBRARY}"
)
target_link_libraries(compression_python PRIVATE
        spdlog::spdlog
)
target_compile_features(compression_python PUBLIC cxx_std_20)
