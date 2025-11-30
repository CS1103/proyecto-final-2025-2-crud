#ifndef PROGRAIII_NN_OPTIMIZER_H
#define PROGRAIII_NN_OPTIMIZER_H
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
#include <unordered_map>
namespace utec {
namespace neural_network {

template<typename T, size_t N>
using Tensor = utec::algebra::Tensor<T, N>;

template<typename T>
class SGD final : public IOptimizer<T> {
private:
    T learning_rate;

public:
    explicit SGD(T learning_rate = 0.01) : learning_rate(learning_rate) {}

    void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) override {
        auto param_it = params.begin();
        auto grad_it = grads.begin();

        while (param_it != params.end()) {
            *param_it = (*param_it) - learning_rate * (*grad_it);
            ++param_it;
            ++grad_it;
        }
    }
};

template<typename T>
class Adam final : public IOptimizer<T> {
private:
    T learning_rate;
    T beta1;
    T beta2;
    T epsilon;
    int t;
    std::unordered_map<void*, utec::algebra::Tensor<T, 2>> m_cache;
    std::unordered_map<void*, utec::algebra::Tensor<T, 2>> v_cache;

public:
    explicit Adam(T learning_rate = 0.001, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {}

    void update(utec::algebra::Tensor<T, 2>& params, const utec::algebra::Tensor<T, 2>& grads) override {
        void* param_ptr = static_cast<void*>(&params);

        if (m_cache.find(param_ptr) == m_cache.end()) {
            auto shape = params.shape();
            size_t rows = shape[0];
            size_t cols = shape[1];

            utec::algebra::Tensor<T, 2> m_tensor(rows, cols);
            utec::algebra::Tensor<T, 2> v_tensor(rows, cols);

            m_tensor.fill(static_cast<T>(0));
            v_tensor.fill(static_cast<T>(0));

            m_cache[param_ptr] = m_tensor;
            v_cache[param_ptr] = v_tensor;
        }

        auto& m = m_cache[param_ptr];
        auto& v = v_cache[param_ptr];

        auto param_it = params.begin();
        auto grad_it = grads.begin();
        auto m_it = m.begin();
        auto v_it = v.begin();

        while (param_it != params.end()) {
            *m_it = beta1 * (*m_it) + (static_cast<T>(1) - beta1) * (*grad_it);
            *v_it = beta2 * (*v_it) + (static_cast<T>(1) - beta2) * (*grad_it) * (*grad_it);

            T m_hat = (*m_it) / (static_cast<T>(1) - std::pow(beta1, t + 1));
            T v_hat = (*v_it) / (static_cast<T>(1) - std::pow(beta2, t + 1));

            *param_it = (*param_it) - learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);

            ++param_it;
            ++grad_it;
            ++m_it;
            ++v_it;
        }
    }

    void step() override {
        t++;
    }
};

}
}
#endif