#include <iostream>
#include <random>
#include <iomanip>
#include "neural_network.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
#include "tensor.h"

using namespace utec::neural_network;


// UTILIDAD PARA TEST
#define TEST_ASSERT(cond, msg) \
    if (!(cond)) std::cout << "[FAIL] " << msg << "\n"; \
    else std::cout << "[PASS] " << msg << "\n";

// ============================================================================
// TEST 1: Tensor Operations
// ============================================================================
void test_tensor_basic() {
    Tensor<double, 2> t1(2, 3);
    t1.fill(1.0);

    Tensor<double, 2> t2(2, 3);
    t2.fill(2.0);

    auto result = t1 + t2;

    TEST_ASSERT(result(0,0) == 3.0, "TensorTest: BasicOperations");
}

// ============================================================================
// TEST 2: Dense Forward Pass
// ============================================================================
void test_dense_forward() {
    Dense<double> layer(
        2, 3,
        [](auto& w){ w.fill(0.5); },
        [](auto& b){ b.fill(0.1); }
    );

    Tensor<double, 2> input(1, 2);
    input(0,0) = 1.0;
    input(0,1) = 2.0;

    auto output = layer.forward(input);

    TEST_ASSERT(std::abs(output(0,0) - 1.6) < 1e-6,
                "DenseTest: ForwardPass");
}

// ============================================================================
// TEST 3: Dense Backward
// ============================================================================
void test_dense_backward() {
    Dense<double> layer(
        2, 1,
        [](auto& w){ w.fill(0.5); },
        [](auto& b){ b.fill(0.0); }
    );

    Tensor<double, 2> input(1, 2);
    input(0,0) = 1.0;
    input(0,1) = 2.0;

    layer.forward(input);

    Tensor<double, 2> grad_out(1, 1);
    grad_out(0,0) = 1.0;

    auto grad_input = layer.backward(grad_out);

    TEST_ASSERT(grad_input.shape()[0] == 1 &&
                grad_input.shape()[1] == 2,
                "DenseTest: Backward");
}

// ============================================================================
// TEST 4: Loss Functions
// ============================================================================
void test_mse_loss() {
    Tensor<double, 2> pred(2, 1);
    pred(0,0) = 0.8;
    pred(1,0) = 0.2;

    Tensor<double, 2> target(2, 1);
    target(0,0) = 1.0;
    target(1,0) = 0.0;

    MSELoss<double> loss(pred, target);
    double lv = loss.loss();

    TEST_ASSERT(std::abs(lv - 0.04) < 1e-6,
                "LossTest: MSELoss");
}

// ============================================================================
// TEST 5: Activation Functions
// ============================================================================
void test_relu() {
    ReLU<double> relu;

    Tensor<double, 2> input(2, 2);
    input(0,0) = -1.0;
    input(0,1) = 2.0;
    input(1,0) = -3.0;
    input(1,1) = 4.0;

    auto output = relu.forward(input);

    bool ok = output(0,0) == 0.0 &&
              output(0,1) == 2.0 &&
              output(1,0) == 0.0 &&
              output(1,1) == 4.0;

    TEST_ASSERT(ok, "ActivationTest: ReLU");
}

// ============================================================================
// TEST 6: Sigmoid Activation
// ============================================================================
void test_sigmoid() {
    Sigmoid<double> sigmoid;

    Tensor<double, 2> input(2, 2);
    input(0,0) = 0.0;
    input(0,1) = 10.0;
    input(1,0) = -10.0;
    input(1,1) = 1.0;

    auto output = sigmoid.forward(input);

    bool ok = std::abs(output(0,0) - 0.5) < 1e-6 &&
              output(0,1) > 0.99 &&
              output(1,0) < 0.01;

    TEST_ASSERT(ok, "ActivationTest: Sigmoid");
}

// ============================================================================
// TEST 7: BCE Loss
// ============================================================================
void test_bce_loss() {
    Tensor<double, 2> pred(2, 1);
    pred(0,0) = 0.9;
    pred(1,0) = 0.1;

    Tensor<double, 2> target(2, 1);
    target(0,0) = 1.0;
    target(1,0) = 0.0;

    BCELoss<double> loss(pred, target);
    double lv = loss.loss();

    TEST_ASSERT(lv < 0.2, "LossTest: BCELoss");
}

// ============================================================================
// DEMO 1: XOR con SGD
// ============================================================================
void demo_xor_sgd() {
    std::cout << "\n========================================\n";
    std::cout << "DEMO 1: XOR Problem con SGD\n";
    std::cout << "========================================\n";

    NeuralNetwork<double> model;

    model.add_layer(std::make_unique<Dense<double>>(
        2, 4,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.5);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        4, 1,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.5);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<Sigmoid<double>>());

    Tensor<double, 2> X(4, 2);
    X = {{0,0},{0,1},{1,0},{1,1}};

    Tensor<double, 2> Y(4, 1);
    Y = {{0},{1},{1},{0}};

    std::cout << "Entrenando...\n";
    model.train<MSELoss, SGD>(X, Y, 2000, 4, 0.5);

    auto pred = model.predict(X);

    std::cout << "\nResultados XOR:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Input -> Target -> Predicción\n";
    std::cout << "(0,0) -> 0 -> " << pred(0,0) << "\n";
    std::cout << "(0,1) -> 1 -> " << pred(1,0) << "\n";
    std::cout << "(1,0) -> 1 -> " << pred(2,0) << "\n";
    std::cout << "(1,1) -> 0 -> " << pred(3,0) << "\n";
}

// ============================================================================
// DEMO 2: XOR con Adam
// ============================================================================
void demo_xor_adam() {
    std::cout << "\n========================================\n";
    std::cout << "DEMO 2: XOR Problem con Adam\n";
    std::cout << "========================================\n";

    NeuralNetwork<double> model;

    model.add_layer(std::make_unique<Dense<double>>(
        2, 8,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.3);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        8, 4,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.3);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        4, 1,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.3);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<Sigmoid<double>>());

    Tensor<double, 2> X(4, 2);
    X = {{0,0},{0,1},{1,0},{1,1}};

    Tensor<double, 2> Y(4, 1);
    Y = {{0},{1},{1},{0}};

    std::cout << "Entrenando con Adam optimizer...\n";
    model.train<BCELoss, Adam>(X, Y, 1000, 4, 0.01);

    auto pred = model.predict(X);

    std::cout << "\nResultados XOR (Adam):\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Input -> Target -> Predicción\n";
    std::cout << "(0,0) -> 0 -> " << pred(0,0) << "\n";
    std::cout << "(0,1) -> 1 -> " << pred(1,0) << "\n";
    std::cout << "(1,0) -> 1 -> " << pred(2,0) << "\n";
    std::cout << "(1,1) -> 0 -> " << pred(3,0) << "\n";
}

// ============================================================================
// DEMO 3: Clasificación Binaria
// ============================================================================
void demo_binary_classification() {
    std::cout << "\n========================================\n";
    std::cout << "DEMO 3: Clasificación Binaria\n";
    std::cout << "========================================\n";

    Tensor<double, 2> X(100, 2);
    Tensor<double, 2> Y(100, 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.2);

    for(size_t i = 0; i < 50; ++i) {
        X(i, 0) = noise(gen);
        X(i, 1) = noise(gen);
        Y(i, 0) = 0.0;
    }

    for(size_t i = 50; i < 100; ++i) {
        X(i, 0) = 1.0 + noise(gen);
        X(i, 1) = 1.0 + noise(gen);
        Y(i, 0) = 1.0;
    }

    NeuralNetwork<double> model;

    model.add_layer(std::make_unique<Dense<double>>(
        2, 8,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.5);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        8, 1,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.5);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<Sigmoid<double>>());

    std::cout << "Entrenando clasificador binario...\n";
    model.train<BCELoss, SGD>(X, Y, 500, 32, 0.1);

    Tensor<double, 2> X_test(4, 2);
    X_test = {{0.0, 0.0}, {1.0, 1.0}, {0.5, 0.5}, {-0.2, -0.2}};

    auto predictions = model.predict(X_test);

    std::cout << "\nPredicciones:\n";
    std::cout << std::fixed << std::setprecision(4);
    for(size_t i = 0; i < 4; ++i) {
        std::cout << "(" << X_test(i,0) << ", " << X_test(i,1)
                  << ") -> " << predictions(i,0)
                  << " (clase " << (predictions(i,0) > 0.5 ? 1 : 0) << ")\n";
    }
}

// ============================================================================
// DEMO 4: Regresión Simple
// ============================================================================
void demo_regression() {
    std::cout << "\n========================================\n";
    std::cout << "DEMO 4: Regresión: y = 2x + 1\n";
    std::cout << "========================================\n";

    Tensor<double, 2> X(50, 1);
    Tensor<double, 2> Y(50, 1);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dist(0.0, 10.0);
    std::normal_distribution<> noise(0.0, 0.5);

    for(size_t i = 0; i < 50; ++i) {
        double x = x_dist(gen);
        X(i, 0) = x;
        Y(i, 0) = 2.0 * x + 1.0 + noise(gen);
    }

    NeuralNetwork<double> model;

    model.add_layer(std::make_unique<Dense<double>>(
        1, 10,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.3);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    model.add_layer(std::make_unique<ReLU<double>>());

    model.add_layer(std::make_unique<Dense<double>>(
        10, 1,
        [](auto& w){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(0, 0.3);
            for(auto& v : w) v = d(gen);
        },
        [](auto& b){ b.fill(0.0); }
    ));

    std::cout << "Entrenando modelo de regresión...\n";
    model.train<MSELoss, Adam>(X, Y, 1000, 32, 0.01);

    Tensor<double, 2> X_test(5, 1);
    X_test = {{1.0}, {2.0}, {3.0}, {4.0}, {5.0}};

    auto predictions = model.predict(X_test);

    std::cout << "\nPredicciones (esperado: y = 2x + 1):\n";
    std::cout << std::fixed << std::setprecision(4);
    for(size_t i = 0; i < 5; ++i) {
        double x = X_test(i, 0);
        double expected = 2.0 * x + 1.0;
        std::cout << "x=" << x << " -> pred=" << predictions(i,0)
                  << " (esperado=" << expected << ")\n";
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "    NEURAL NETWORK FRAMEWORK TEST\n";
    std::cout << "========================================\n\n";

    std::cout << "=== Running Unit Tests ===\n\n";
    test_tensor_basic();
    test_dense_forward();
    test_dense_backward();
    test_mse_loss();
    test_relu();
    test_sigmoid();
    test_bce_loss();

    demo_xor_sgd();
    demo_xor_adam();
    demo_binary_classification();
    demo_regression();

    std::cout << "\n========================================\n";
    std::cout << "    TODAS LAS PRUEBAS COMPLETADAS\n";
    std::cout << "========================================\n";

    return 0;
}



