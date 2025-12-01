#ifndef PROGRAIII_NEURAL_NETWORK_H
#define PROGRAIII_NEURAL_NETWORK_H
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <forward_list>
#include <string>
#include <numeric>
#include <fstream>
#include <map>
#include <set>
#include <memory>
#include <initializer_list>
#include <stdexcept>
#include <array>
#include <chrono>
#include <cstddef>
#include <type_traits>
#include <thread>
#include <functional>
#include <random>
#include "nn_interfaces.h"
#include "tensor.h"
#include "nn_dense.h"
#include "nn_activation.h"
#include "nn_loss.h"
#include "nn_optimizer.h"
namespace utec {
    namespace neural_network {

        template<typename T, size_t N>
        using Tensor = utec::algebra::Tensor<T, N>;

        template<typename T>
        class NeuralNetwork {
        private:
            std::vector<std::unique_ptr<ILayer<T>>> layers;

        public:
            void add_layer(std::unique_ptr<ILayer<T>> layer) {
                layers.push_back(std::move(layer));
            }

            template<template<typename...> class LossType,
                     template<typename...> class OptimizerType = SGD>
            void train(const utec::algebra::Tensor<T, 2>& X, const utec::algebra::Tensor<T, 2>& Y,
                       const size_t epochs, const size_t batch_size, T learning_rate,
                       std::function<void(size_t,double,double,const utec::algebra::Tensor<T,2>&, T)> on_epoch_end = nullptr) {

                OptimizerType<T> optimizer(learning_rate);
                double elapsed_total = 0.0;

                for (size_t epoch = 0; epoch < epochs; ++epoch) {
                    auto epoch_t0 = std::chrono::steady_clock::now();

                    utec::algebra::Tensor<T, 2> output = X;

                    for (auto& layer : layers) {
                        output = layer->forward(output);
                    }

                    LossType<T> loss_fn(output, Y);
                    T loss_value = loss_fn.loss();

                    utec::algebra::Tensor<T, 2> gradient = loss_fn.loss_gradient();

                    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                        gradient = (*it)->backward(gradient);
                    }

                    for (auto& layer : layers) {
                        layer->update_params(optimizer);
                    }

                    optimizer.step();

                    auto epoch_t1 = std::chrono::steady_clock::now();
                    double elapsed_epoch = std::chrono::duration<double>(epoch_t1 - epoch_t0).count();
                    elapsed_total += elapsed_epoch;

                    if (on_epoch_end) {
                        on_epoch_end(epoch + 1, elapsed_epoch, elapsed_total, output, loss_value);
                    }
                }
            }

            utec::algebra::Tensor<T, 2> predict(const utec::algebra::Tensor<T, 2>& X) {
                utec::algebra::Tensor<T, 2> output = X;

                for (auto& layer : layers) {
                    output = layer->forward(output);
                }

                return output;
            }
        };

    }
}

#endif //PROGRAIII_NEURAL_NETWORK_H