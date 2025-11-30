#include "neural_network.h"
#include "tensor.h"
using namespace utec::neural_network;

int main() {
    // Datos XOR
    Tensor<double, 2> X(4, 2);
    X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Tensor<double, 2> Y(4, 1);
    Y = {0, 1, 1, 0};
    
    // Crear modelo
    NeuralNetwork<double> model;
    
    // Arquitectura: 2 -> 4 -> 1
    model.add_layer(std::make_unique<Dense<double>>(
        2, 4,
        [](auto& w) { 
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dist(0.0, 0.5);
            for(auto& val : w) val = dist(gen);
        },
        [](auto& b) { b.fill(0.0); }
    ));
    
    model.add_layer(std::make_unique<ReLU<double>>());
    
    model.add_layer(std::make_unique<Dense<double>>(
        4, 1,
        [](auto& w) { /* inicialización */ },
        [](auto& b) { b.fill(0.0); }
    ));
    
    model.add_layer(std::make_unique<Sigmoid<double>>());
    
    // Entrenar
    model.train<MSELoss, Adam>(X, Y, 5000, 4, 0.01);
    
    // Predecir
    auto predictions = model.predict(X);
    std::cout << "Predicciones:\n" << predictions;
    
    return 0;
}