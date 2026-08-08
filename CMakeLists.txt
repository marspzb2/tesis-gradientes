cmake_minimum_required(VERSION 3.30)

project(entrenamiento LANGUAGES CXX)

# C++ flags for module support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# FILE_SET CXX_MODULES enables scanning for module sources automatically.
# Keeping it off globally avoids scanning third-party targets and try_compile().
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

#Dependencies
include(FetchContent)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/third-party")

    file(DOWNLOAD
        https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.9.1.zip
        "${CMAKE_BINARY_DIR}/third-party/libtorch.zip"
    )

    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E tar xzf libtorch.zip
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/third-party"
        COMMAND_ERROR_IS_FATAL ANY
    )

    list(APPEND CMAKE_PREFIX_PATH
        "${CMAKE_BINARY_DIR}/third-party/libtorch"
    )
endif()

find_package(Torch REQUIRED CONFIG)


FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# Subdirectories
add_subdirectory(src/cpp/compression)
add_subdirectory(src/cpp/utils)

add_executable(entrenamiento src/cpp/main.cpp)
target_compile_features(entrenamiento PRIVATE cxx_std_20)
set_property(TARGET entrenamiento PROPERTY CXX_SCAN_FOR_MODULES ON)

# Linking
target_link_libraries(entrenamiento
        PRIVATE
        compression-base
        utils
)

target_link_libraries(entrenamiento PRIVATE torch)
