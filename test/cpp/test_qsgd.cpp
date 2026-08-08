#include <gtest/gtest.h>
#include <torch/torch.h>

// Test básico para verificar que LibTorch crea tensores correctamente
TEST(PyTorchTest, TensorCreation) {
    // Crea una matriz de 2x3 con números aleatorios
    torch::Tensor t = torch::rand({2, 3});
    
    // Verifica que las dimensiones sean correctas
    EXPECT_EQ(t.dim(), 2);
    EXPECT_EQ(t.size(0), 2);
    EXPECT_EQ(t.size(1), 3);
}

TEST(PyTorchTest, CUDA_Availability) {
    // Este test pasa siempre, solo imprime si tenés soporte GPU activo
    if (torch::cuda::is_available()) {
        SUCCEED() << "CUDA está disponible en este entorno de ejecución.";
    } else {
        std::cout << "[ INFO ] Ejecutando tests únicamente en CPU." << std::endl;
    }
}