#ifndef PROGRAIII_NN_DENSE_H
#define PROGRAIII_NN_DENSE_H
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
#include "nn_interfaces.h"
#include "tensor.h"
namespace utec {
namespace neural_network {

template<typename T, size_t N>
using Tensor = utec::algebra::Tensor<T, N>;

template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T, 2> weights;
    utec::algebra::Tensor<T, 2> bias;
    utec::algebra::Tensor<T, 2> input_cache;
    utec::algebra::Tensor<T, 2> dW;
    utec::algebra::Tensor<T, 2> db;

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_f, size_t out_f, InitWFun init_w_fun, InitBFun init_b_fun)
        : weights(in_f, out_f),
          bias(size_t(1), out_f),
          dW(in_f, out_f),
          db(size_t(1), out_f) {

        init_w_fun(weights);
        init_b_fun(bias);

        dW.fill(static_cast<T>(0));
        db.fill(static_cast<T>(0));
    }

    utec::algebra::Tensor<T, 2> forward(const utec::algebra::Tensor<T, 2>& x) override {
        input_cache = x;

        auto x_shape = x.shape();
        auto w_shape = weights.shape();
        size_t batch_size = x_shape[0];
        size_t out_features = w_shape[1];

        utec::algebra::Tensor<T, 2> result(batch_size, out_features);

        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < w_shape[0]; ++k) {
                    sum += x(i, k) * weights(k, j);
                }
                result(i, j) = sum + bias(0, j);
            }
        }

        return result;
    }

    utec::algebra::Tensor<T, 2> backward(const utec::algebra::Tensor<T, 2>& dZ) override {
        auto dZ_shape = dZ.shape();
        auto x_shape = input_cache.shape();
        auto w_shape = weights.shape();

        size_t batch_size = dZ_shape[0];
        size_t out_features = dZ_shape[1];
        size_t in_features = w_shape[0];

        for (size_t i = 0; i < in_features; ++i) {
            for (size_t j = 0; j < out_features; ++j) {
                T sum = static_cast<T>(0);
                for (size_t b = 0; b < batch_size; ++b) {
                    sum += input_cache(b, i) * dZ(b, j);
                }
                dW(i, j) = sum;
            }
        }

        for (size_t j = 0; j < out_features; ++j) {
            T sum = static_cast<T>(0);
            for (size_t b = 0; b < batch_size; ++b) {
                sum += dZ(b, j);
            }
            db(0, j) = sum;
        }

        utec::algebra::Tensor<T, 2> dX(batch_size, in_features);

        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < in_features; ++j) {
                T sum = static_cast<T>(0);
                for (size_t k = 0; k < out_features; ++k) {
                    sum += dZ(i, k) * weights(j, k);
                }
                dX(i, j) = sum;
            }
        }

        return dX;
    }

    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights, dW);
        optimizer.update(bias, db);
    }
};

}
}
#endif