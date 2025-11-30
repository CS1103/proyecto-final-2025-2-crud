#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H
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
#include <cmath>
namespace utec {
namespace algebra {
    template<typename T, size_t N, typename Func>
    Tensor<T, N> apply(const Tensor<T, N>& tensor, Func func) {
        Tensor<T, N> result = tensor;
        for (auto& val : result) {
            val = func(val);
        }
        return result;
    }
}

namespace neural_network {

template<typename T, size_t N>
using Tensor = utec::algebra::Tensor<T, N>;

template<typename T>
class ReLU final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> input_cache;

public:
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        input_cache = z;

        auto shape = z.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> result(rows, cols);

        auto z_it = z.begin();
        auto result_it = result.begin();

        while (z_it != z.end()) {
            *result_it = (*z_it > static_cast<T>(0)) ? *z_it : static_cast<T>(0);
            ++z_it;
            ++result_it;
        }

        return result;
    }

    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& g) override {
        auto shape = g.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> result(rows, cols);

        auto g_it = g.begin();
        auto input_it = input_cache.begin();
        auto result_it = result.begin();

        while (g_it != g.end()) {
            *result_it = (*input_it > static_cast<T>(0)) ? *g_it : static_cast<T>(0);
            ++g_it;
            ++input_it;
            ++result_it;
        }

        return result;
    }
};

template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> output_cache;

public:
    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& z) override {
        auto shape = z.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> result(rows, cols);

        auto z_it = z.begin();
        auto result_it = result.begin();

        while (z_it != z.end()) {
            *result_it = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-(*z_it)));
            ++z_it;
            ++result_it;
        }

        output_cache = result;
        return result;
    }

    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& g) override {
        auto shape = g.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> result(rows, cols);

        auto g_it = g.begin();
        auto output_it = output_cache.begin();
        auto result_it = result.begin();

        while (g_it != g.end()) {
            *result_it = (*g_it) * (*output_it) * (static_cast<T>(1) - (*output_it));
            ++g_it;
            ++output_it;
            ++result_it;
        }

        return result;
    }
};

}
}
#endif