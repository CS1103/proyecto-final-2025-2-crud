#include "tensor.h"
#include <random>    
#include <memory>   
#include <cstddef>   
#include <iostream>
#include "neural_network.h"

// Datos de regresión: y = 2x + 1 + ruido
Tensor<double, 2> X(100, 1);
Tensor<double, 2> Y(100, 1);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> x_dist(0.0, 10.0);
std::normal_distribution<> noise(0.0, 0.5);

for(size_t i = 0; i < 100; ++i) {
    double x = x_dist(gen);
    X(i, 0) = x;
    Y(i, 0) = 2.0 * x + 1.0 + noise(gen);
}

// Modelo simple: 1 -> 1 (sin activación)
NeuralNetwork<double> model;
model.add_layer(std::make_unique<Dense<double>>(1, 1, ...));

model.train<MSELoss, SGD>(X, Y, 1000, 32, 0.01);

// Predecir para nuevos valores
Tensor<double, 2> X_test(5, 1);
X_test = {1.0, 2.0, 3.0, 4.0, 5.0};
auto predictions = model.predict(X_test);