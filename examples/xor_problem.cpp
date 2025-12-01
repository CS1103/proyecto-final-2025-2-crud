#include "neural_network.h"
#include "tensor.h"
#include <fstream>
#include <chrono>
#include <sstream>

using namespace utec::neural_network;

int main(int argc, char** argv) {
    // Parseo simple de argumentos
    std::string optimizer = "Adam";
    double lr = 0.01;
    size_t epochs = 5000;
    size_t batch = 4;
    int seed = 42;
    std::string out = "logs/xor_" + optimizer + "_seed" + std::to_string(seed) + ".csv";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--optimizer" && i+1<argc) optimizer = argv[++i];
        else if (a == "--lr" && i+1<argc) lr = std::stod(argv[++i]);
        else if (a == "--epochs" && i+1<argc) epochs = std::stoul(argv[++i]);
        else if (a == "--batch" && i+1<argc) batch = std::stoul(argv[++i]);
        else if (a == "--seed" && i+1<argc) seed = std::stoi(argv[++i]);
        else if (a == "--out" && i+1<argc) out = argv[++i];
    }

    // Datos XOR
    std::mt19937 gen(seed);
    Tensor<double, 2> X(4, 2);
    X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    Tensor<double, 2> Y(4, 1);
    Y = {0, 1, 1, 0};

    // Crear modelo
    NeuralNetwork<double> model;

    // Arquitectura: 2 -> 4 -> 1
    model.add_layer(std::make_unique<Dense<double>>(
        2, 4,
        [&](auto& w) {
            std::normal_distribution<> dist(0.0, 0.5);
            for(auto& val : w) val = dist(gen);
        },
        [&](auto& b) { b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        4, 1,
        [&](auto& w) { std::normal_distribution<> dist(0.0, 0.5); for(auto& val : w) val = dist(gen); },
        [&](auto& b) { b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<Sigmoid<double>>());

    // Preparar archivo CSV
    std::ofstream outfs(out);
    outfs << "epoch,elapsed_epoch_sec,elapsed_total_sec,train_loss,val_loss,train_acc,val_acc,optimizer,lr,batch_size,seed,notes\n";

    // Callback para on_epoch_end
    auto on_epoch = [&](size_t epoch, double elapsed_epoch, double elapsed_total, const Tensor<double,2>& output, double loss_value){
        // calcular accuracy simple
        size_t correct = 0;
        for (size_t i = 0; i < 4; ++i) {
            double pred = output(i,0) > 0.5 ? 1.0 : 0.0;
            if (std::abs(pred - Y(i,0)) < 1e-9) ++correct;
        }
        double acc = static_cast<double>(correct) / 4.0;
        // escribimos línea CSV
        outfs << epoch << "," << elapsed_epoch << "," << elapsed_total << "," << loss_value << "," << loss_value << "," << acc << "," << acc << "," << optimizer << "," << lr << "," << batch << "," << seed << "," << "" << "\n";
        outfs.flush();
    };

    // Ejecutar entrenamiento con callback
    if (optimizer == "SGD") {
        model.train<MSELoss, SGD>(X, Y, epochs, batch, lr, on_epoch);
    } else {
        model.train<MSELoss, Adam>(X, Y, epochs, batch, lr, on_epoch);
    }

    outfs.close();

    // Predecir final y mostrar
    auto predictions = model.predict(X);
    std::cout << "Predicciones:\n" << predictions;

    return 0;
}