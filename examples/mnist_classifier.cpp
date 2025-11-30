#include "tensor.h"
#include <random>    
#include <memory>   
#include <cstddef>   
#include <iostream>
#include "neural_network.h"

// Generar datos sintéticos
Tensor<double, 2> X(100, 2);
Tensor<double, 2> Y(100, 1);

// Clase 0: puntos alrededor de (0, 0)
// Clase 1: puntos alrededor de (1, 1)
std::random_device rd;
std::mt19937 gen(rd());
std::normal_distribution<> dist(0.0, 0.2);

for(size_t i = 0; i < 50; ++i) {
    X(i, 0) = dist(gen);
    X(i, 1) = dist(gen);
    Y(i, 0) = 0.0;
}

for(size_t i = 50; i < 100; ++i) {
    X(i, 0) = 1.0 + dist(gen);
    X(i, 1) = 1.0 + dist(gen);
    Y(i, 0) = 1.0;
}

// Crear y entrenar modelo
NeuralNetwork<double> model;
model.add_layer(std::make_unique<Dense<double>>(2, 8, ...));
model.add_layer(std::make_unique<ReLU<double>>());
model.add_layer(std::make_unique<Dense<double>>(8, 1, ...));
model.add_layer(std::make_unique<Sigmoid<double>>());

model.train<BCELoss, SGD>(X, Y, 1000, 32, 0.1);