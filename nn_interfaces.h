#ifndef PROGRAIII_NN_INTERFACES_H
#define PROGRAIII_NN_INTERFACES_H
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
#include <initializer_list>
#include <stdexcept>
#include <array>
#include <chrono>
#include <cstddef>
#include <type_traits>
#include <thread>
#include <functional>
#include <random>
#include "tensor.h"
namespace utec {
namespace neural_network {
        template<typename T>
        class IOptimizer;

        template<typename T>
        class ILayer {
        public:
            virtual ~ILayer() = default;

            virtual utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& input) = 0;

            virtual utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& gradient) = 0;

            virtual void update_params(IOptimizer<T>& optimizer) {
            }
        };

        template<typename T, size_t DIMS>
        class ILoss {
        public:
            virtual ~ILoss() = default;

            virtual T loss() const = 0;

            virtual utec::algebra::Tensor<T, DIMS> loss_gradient() const = 0;
        };

        template<typename T>
        class IOptimizer {
        public:
            virtual ~IOptimizer() = default;

            virtual void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) = 0;

            virtual void step() {
            }
        };

    }
}

#endif //PROGRAIII_NN_INTERFACES_H