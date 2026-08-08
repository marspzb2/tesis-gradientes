cmake_minimum_required(VERSION 4.2)
project(entrenamiento_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Habilitar el motor de pruebas de CMake
enable_testing()

# 2. Buscar los paquetes que Conan ya instaló y preparó
find_package(GTest REQUIRED)
find_package(libtorch REQUIRED) # Conan usa 'libtorch', no 'Torch'

# 3. Definir el ejecutable de los tests
add_executable(run_tests
        test_qsgd.cpp
)

# 4. Vincular las librerías usando los targets generados por Conan
target_link_libraries(run_tests
        PRIVATE
        GTest::gtest_main
        libtorch::libtorch
)

# 5. Descubrir automáticamente las pruebas para CTest
include(GoogleTest)
gtest_discover_tests(run_tests)