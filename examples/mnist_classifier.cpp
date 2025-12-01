#include "tensor.h"
#include <random>
#include <memory>
#include <cstddef>
#include <iostream>
#include "neural_network.h"
#include <fstream>
#include <chrono>
#include <string>

using namespace utec::neural_network;

int main(int argc, char** argv) {
    // argumentos por defecto
    std::string optimizer = "SGD";
    double lr = 0.1;
    size_t epochs = 1000;
    size_t batch = 32;
    int seed = 123;
    std::string out = "logs/mnist_" + optimizer + "_seed" + std::to_string(seed) + ".csv";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--optimizer" && i+1<argc) optimizer = argv[++i];
        else if (a == "--lr" && i+1<argc) lr = std::stod(argv[++i]);
        else if (a == "--epochs" && i+1<argc) epochs = std::stoul(argv[++i]);
        else if (a == "--batch" && i+1<argc) batch = std::stoul(argv[++i]);
        else if (a == "--seed" && i+1<argc) seed = std::stoi(argv[++i]);
        else if (a == "--out" && i+1<argc) out = argv[++i];
    }

    // Generar datos sintéticos
    Tensor<double, 2> X(100, 2);
    Tensor<double, 2> Y(100, 1);

    std::mt19937 gen(seed);
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
    model.add_layer(std::make_unique<Dense<double>>(2, 8,
        [&](auto& w) {
            std::normal_distribution<> d(0.0, 0.5);
            for (auto& v : w) v = d(gen);
        },
        [&](auto& b) { b.fill(0.0); }
    ));
    model.add_layer(std::make_unique<ReLU<double>>());
    model.add_layer(std::make_unique<Dense<double>>(8, 1,
        [&](auto& w) {
            std::normal_distribution<> d(0.0, 0.5);
            for (auto& v : w) v = d(gen);
        },
        [&](auto& b) { b.fill(0.0); }
    ));
    model.add_layer(std::make_unique<Sigmoid<double>>());

    // Preparar archivo CSV
    std::ofstream outfs(out);
    outfs << "epoch,elapsed_epoch_sec,elapsed_total_sec,train_loss,val_loss,train_acc,val_acc,optimizer,lr,batch_size,seed,notes\n";

    // Callback
    auto on_epoch = [&](size_t epoch, double elapsed_epoch, double elapsed_total, const Tensor<double,2>& output, double loss_value){
        // calcular accuracy en entrenamiento (mismo conjunto)
        size_t correct = 0;
        for (size_t i = 0; i < 100; ++i) {
            double pred = output(i,0) > 0.5 ? 1.0 : 0.0;
            if (std::abs(pred - Y(i,0)) < 1e-9) ++correct;
        }
        double acc = static_cast<double>(correct) / 100.0;
        // escribir CSV
        outfs << epoch << "," << elapsed_epoch << "," << elapsed_total << "," << loss_value << "," << loss_value << "," << acc << "," << acc << "," << optimizer << "," << lr << "," << batch << "," << seed << "," << "" << "\n";
        outfs.flush();
    };

    if (optimizer == "SGD") {
        model.train<BCELoss, SGD>(X, Y, epochs, batch, lr, on_epoch);
    } else {
        model.train<BCELoss, Adam>(X, Y, epochs, batch, lr, on_epoch);
    }

    outfs.close();

    auto output = model.predict(X);
    size_t correct = 0;
    for (size_t i = 0; i < 100; ++i) {
        double pred = output(i,0) > 0.5 ? 1.0 : 0.0;
        if (std::abs(pred - Y(i,0)) < 1e-9) ++correct;
    }
    std::cout << "Final accuracy: " << static_cast<double>(correct)/100.0 << std::endl;

    return 0;
}
