#ifndef PROGRAIII_NN_LOSS_H
#define PROGRAIII_NN_LOSS_H
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
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_prediction;
    utec::algebra::Tensor<T, 2> y_true;

public:
    MSELoss(const utec::algebra::Tensor<T, 2>& y_prediction, const utec::algebra::Tensor<T, 2>& y_true)
        : y_prediction(y_prediction), y_true(y_true) {
    }

    T loss() const override {
        T sum = static_cast<T>(0);
        size_t n = y_prediction.size();

        auto pred_it = y_prediction.begin();
        auto true_it = y_true.begin();

        while (pred_it != y_prediction.end()) {
            T diff = (*pred_it) - (*true_it);
            sum += diff * diff;
            ++pred_it;
            ++true_it;
        }

        return sum / static_cast<T>(n);
    }

    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto shape = y_prediction.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> gradient(rows, cols);

        size_t n = y_prediction.size();
        T factor = static_cast<T>(2) / static_cast<T>(n);

        auto pred_it = y_prediction.begin();
        auto true_it = y_true.begin();
        auto grad_it = gradient.begin();

        while (pred_it != y_prediction.end()) {
            *grad_it = factor * ((*pred_it) - (*true_it));
            ++pred_it;
            ++true_it;
            ++grad_it;
        }

        return gradient;
    }
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T, 2> y_prediction;
    utec::algebra::Tensor<T, 2> y_true;
    static constexpr T epsilon = static_cast<T>(1e-7);

public:
    BCELoss(const utec::algebra::Tensor<T, 2>& y_prediction, const utec::algebra::Tensor<T, 2>& y_true)
        : y_prediction(y_prediction), y_true(y_true) {
    }

    T loss() const override {
        T sum = static_cast<T>(0);
        size_t n = y_prediction.size();

        auto pred_it = y_prediction.begin();
        auto true_it = y_true.begin();

        while (pred_it != y_prediction.end()) {
            T pred = *pred_it;
            T true_val = *true_it;

            pred = std::max(epsilon, std::min(static_cast<T>(1) - epsilon, pred));

            sum += -(true_val * std::log(pred) + (static_cast<T>(1) - true_val) * std::log(static_cast<T>(1) - pred));

            ++pred_it;
            ++true_it;
        }

        return sum / static_cast<T>(n);
    }

    utec::algebra::Tensor<T, 2> loss_gradient() const override {
        auto shape = y_prediction.shape();
        size_t rows = shape[0];
        size_t cols = shape[1];

        utec::algebra::Tensor<T, 2> gradient(rows, cols);

        size_t n = y_prediction.size();

        auto pred_it = y_prediction.begin();
        auto true_it = y_true.begin();
        auto grad_it = gradient.begin();

        while (pred_it != y_prediction.end()) {
            T pred = *pred_it;
            T true_val = *true_it;

            pred = std::max(epsilon, std::min(static_cast<T>(1) - epsilon, pred));

            *grad_it = (-(true_val / pred - (static_cast<T>(1) - true_val) / (static_cast<T>(1) - pred))) / static_cast<T>(n);

            ++pred_it;
            ++true_it;
            ++grad_it;
        }

        return gradient;
    }
};

}
}
#endif