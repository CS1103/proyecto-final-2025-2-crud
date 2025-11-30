#ifndef PROGRAIII_TENSOR_H
#define PROGRAIII_TENSOR_H
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
#include <cstddef>
#include <type_traits>

using namespace std;

namespace utec {
namespace algebra {

template<typename T, size_t N>
class Tensor {
    array<size_t, N> dimensions;
    vector<T> data;

    template<typename... Args>
    size_t calculate_size(Args... args) {
        size_t arr[] = { static_cast<size_t>(args)... };
        size_t total = 1;
        for (size_t v : arr) total *= v;
        return total;
    }

    template<typename... Args>
    size_t calculate_index(Args... indices) const {
        size_t arr[] = { static_cast<size_t>(indices)... };
        size_t index = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            index += arr[i] * multiplier;
            multiplier *= dimensions[i];
        }
        return index;
    }

    bool are_broadcastable(const Tensor& other) const {
        for (size_t i = 0; i < N; ++i) {
            if (dimensions[i] != other.dimensions[i] &&
                dimensions[i] != 1 && other.dimensions[i] != 1) {
                return false;
            }
        }
        return true;
    }

    array<size_t, N> broadcast_shape(const Tensor& other) const {
        array<size_t, N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = max(dimensions[i], other.dimensions[i]);
        }
        return result;
    }

    size_t calculate_shape_size(const array<size_t, N>& shape) const {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) total *= shape[i];
        return total;
    }

    size_t get_broadcast_index(size_t linear_idx, const array<size_t, N>& result_shape) const {
        array<size_t, N> indices;
        size_t temp = linear_idx;

        for (int i = N - 1; i >= 0; --i) {
            indices[i] = temp % result_shape[i];
            temp /= result_shape[i];
        }
        for (size_t i = 0; i < N; ++i) {
            if (dimensions[i] == 1) indices[i] = 0;
        }

        size_t result = 0;
        size_t multiplier = 1;
        for (int i = N - 1; i >= 0; --i) {
            result += indices[i] * multiplier;
            multiplier *= dimensions[i];
        }
        return result;
    }

public:
    // Constructor con array<size_t, N>
    Tensor(const array<size_t, N>& dims) : dimensions(dims) {
        size_t total = 1;
        for (size_t i = 0; i < N; ++i) total *= dims[i];
        data.resize(total);
    }
    
    // Constructor por defecto - crea un tensor con dimensiones de tamaño 1
    Tensor() {
        for (size_t i = 0; i < N; ++i) {
            dimensions[i] = 1;
        }
        data.resize(1);
    }
    
    // Constructor con initializer_list de size_t para las dimensiones
    Tensor(initializer_list<size_t> dims) {
        if (dims.size() != N) {
            throw invalid_argument(
                "Number of dimensions do not match with " + to_string(N)
            );
        }
        size_t i = 0;
        for (auto d : dims) {
            dimensions[i++] = d;
        }
        size_t total = 1;
        for (size_t j = 0; j < N; ++j) total *= dimensions[j];
        data.resize(total);
    }

    // Constructor variádico para dimensiones individuales
    template<typename... Args>
    Tensor(Args... args) {
        static_assert(sizeof...(Args) == N, "Number of dimensions must match template parameter N");
        size_t arr[] = { static_cast<size_t>(args)... };
        for (size_t i = 0; i < N; ++i) dimensions[i] = arr[i];
        data.resize(calculate_size(args...));
    }

    // Constructor para inicialización 2D con valores
    Tensor(initializer_list<initializer_list<T>> init) {
        if (N != 2) throw invalid_argument("2D initializer list requires N=2");
        dimensions[0] = init.size();
        dimensions[1] = init.begin()->size();
        data.reserve(dimensions[0] * dimensions[1]);
        for (const auto& row : init)
            for (const auto& val : row) data.push_back(val);
    }

    // Constructor para inicialización 3D con valores
    Tensor(initializer_list<initializer_list<initializer_list<T>>> init) {
        if (N != 3) throw invalid_argument("3D initializer list requires N=3");
        dimensions[0] = init.size();
        dimensions[1] = init.begin()->size();
        dimensions[2] = init.begin()->begin()->size();
        data.reserve(dimensions[0] * dimensions[1] * dimensions[2]);
        for (const auto& matrix : init)
            for (const auto& row : matrix)
                for (const auto& val : row) data.push_back(val);
    }

    array<size_t, N> shape() const { return dimensions; }
    size_t size() const { return data.size(); }

    void fill(const T& value) { std::fill(data.begin(), data.end(), value); }

    Tensor& operator=(initializer_list<T> ilist) {
        if (ilist.size() != data.size())
            throw invalid_argument("Data size does not match tensor size");
        copy(ilist.begin(), ilist.end(), data.begin());
        return *this;
    }

    template<size_t M = N, typename U = T>
    typename enable_if<M == 2, Tensor&>::type
    operator=(initializer_list<initializer_list<U>> ilist) {
        size_t rows = ilist.size();
        size_t cols = ilist.begin()->size();
        if (rows != dimensions[0] || cols != dimensions[1])
            throw invalid_argument("Data size does not match tensor size");
        size_t idx = 0;
        for (const auto& row : ilist) {
            if (row.size() != cols) throw invalid_argument("Inconsistent row sizes");
            for (const auto& val : row) data[idx++] = val;
        }
        return *this;
    }

    template<size_t M = N, typename U = T>
    typename enable_if<M == 3, Tensor&>::type
    operator=(initializer_list<initializer_list<initializer_list<U>>> ilist) {
        size_t dim0 = ilist.size();
        size_t dim1 = ilist.begin()->size();
        size_t dim2 = ilist.begin()->begin()->size();
        if (dim0 != dimensions[0] || dim1 != dimensions[1] || dim2 != dimensions[2])
            throw invalid_argument("Data size does not match tensor size");
        size_t idx = 0;
        for (const auto& matrix : ilist)
            for (const auto& row : matrix)
                for (const auto& val : row) data[idx++] = val;
        return *this;
    }

    template<typename... Args>
    T& operator()(Args... indices) {
        if (sizeof...(Args) != N)
            throw invalid_argument("Number of indices does not match tensor dimensions");
        return data[calculate_index(indices...)];
    }

    template<typename... Args>
    const T& operator()(Args... indices) const {
        if (sizeof...(Args) != N)
            throw invalid_argument("Number of indices does not match tensor dimensions");
        return data[calculate_index(indices...)];
    }

    Tensor& operator+=(const Tensor& other) {
        if (dimensions == other.dimensions) {
            for (size_t i = 0; i < data.size(); ++i) data[i] += other.data[i];
        } else if (are_broadcastable(other)) {
            auto result_shape = broadcast_shape(other);
            size_t total_size = calculate_shape_size(result_shape);
            vector<T> new_data(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                size_t idx1 = get_broadcast_index(i, result_shape);
                size_t idx2 = other.get_broadcast_index(i, result_shape);
                new_data[i] = data[idx1] + other.data[idx2];
            }
            data = move(new_data);
            dimensions = result_shape;
        } else {
            throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        if (dimensions == other.dimensions) {
            for (size_t i = 0; i < data.size(); ++i) data[i] -= other.data[i];
        } else if (are_broadcastable(other)) {
            auto result_shape = broadcast_shape(other);
            size_t total_size = calculate_shape_size(result_shape);
            vector<T> new_data(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                size_t idx1 = get_broadcast_index(i, result_shape);
                size_t idx2 = other.get_broadcast_index(i, result_shape);
                new_data[i] = data[idx1] - other.data[idx2];
            }
            data = move(new_data);
            dimensions = result_shape;
        } else {
            throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return *this;
    }

    Tensor& operator*=(const Tensor& other) {
        if (dimensions == other.dimensions) {
            for (size_t i = 0; i < data.size(); ++i) data[i] *= other.data[i];
        } else if (are_broadcastable(other)) {
            auto result_shape = broadcast_shape(other);
            size_t total_size = calculate_shape_size(result_shape);
            vector<T> new_data(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                size_t idx1 = get_broadcast_index(i, result_shape);
                size_t idx2 = other.get_broadcast_index(i, result_shape);
                new_data[i] = data[idx1] * other.data[idx2];
            }
            data = move(new_data);
            dimensions = result_shape;
        } else {
            throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return *this;
    }

    Tensor& operator/=(const Tensor& other) {
        if (dimensions == other.dimensions) {
            for (size_t i = 0; i < data.size(); ++i) data[i] /= other.data[i];
        } else if (are_broadcastable(other)) {
            auto result_shape = broadcast_shape(other);
            size_t total_size = calculate_shape_size(result_shape);
            vector<T> new_data(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                size_t idx1 = get_broadcast_index(i, result_shape);
                size_t idx2 = other.get_broadcast_index(i, result_shape);
                new_data[i] = data[idx1] / other.data[idx2];
            }
            data = move(new_data);
            dimensions = result_shape;
        } else {
            throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
        }
        return *this;
    }

    Tensor operator+(const Tensor& other) const { Tensor r=*this; r+=other; return r; }
    Tensor operator-(const Tensor& other) const { Tensor r=*this; r-=other; return r; }
    Tensor operator*(const Tensor& other) const { Tensor r=*this; r*=other; return r; }
    Tensor operator/(const Tensor& other) const { Tensor r=*this; r/=other; return r; }

    Tensor operator+(const T& scalar) const { Tensor r=*this; for(auto& v:r.data) v+=scalar; return r; }
    Tensor operator-(const T& scalar) const { Tensor r=*this; for(auto& v:r.data) v-=scalar; return r; }
    Tensor operator*(const T& scalar) const { Tensor r=*this; for(auto& v:r.data) v*=scalar; return r; }
    Tensor operator/(const T& scalar) const { Tensor r=*this; for(auto& v:r.data) v/=scalar; return r; }

    Tensor& operator+=(const T& scalar) { for(auto& v:data) v+=scalar; return *this; }
    Tensor& operator-=(const T& scalar) { for(auto& v:data) v-=scalar; return *this; }
    Tensor& operator*=(const T& scalar) { for(auto& v:data) v*=scalar; return *this; }
    Tensor& operator/=(const T& scalar) { for(auto& v:data) v/=scalar; return *this; }

    friend Tensor operator+(const T& scalar, const Tensor& tensor) { return tensor + scalar; }
    friend Tensor operator-(const T& scalar, const Tensor& tensor) {
        Tensor r = tensor; for (auto& v : r.data) v = scalar - v; return r;
    }
    friend Tensor operator*(const T& scalar, const Tensor& tensor) { return tensor * scalar; }
    friend Tensor operator/(const T& scalar, const Tensor& tensor) {
        Tensor r = tensor; for (auto& v : r.data) v = scalar / v; return r;
    }

    template<typename... Args>
    void reshape(Args... args) {
        if (sizeof...(args) != N) {
            throw invalid_argument(
                "Number of dimensions do not match with " + to_string(N)
            );
        }
        size_t arr[] = { static_cast<size_t>(args)... };

        size_t new_size = 1;
        for (size_t i = 0; i < sizeof...(Args); ++i) new_size *= arr[i];

        if (new_size < data.size()) data.resize(new_size);
        else if (new_size > data.size()) data.resize(new_size, T{});

        for (size_t i = 0; i < N; ++i) dimensions[i] = arr[i];
    }

    auto begin() { return data.begin(); }
    auto end() { return data.end(); }
    auto begin() const { return data.begin(); }
    auto end() const { return data.end(); }
    auto cbegin() const { return data.cbegin(); }
    auto cend() const { return data.cend(); }

    friend ostream& operator<<(ostream& os, const Tensor& t) {
        if (N == 4) {
            os << "{\n";
            for (size_t i = 0; i < t.dimensions[0]; ++i) {
                os << "{\n";
                for (size_t j = 0; j < t.dimensions[1]; ++j) {
                    os << "{\n";
                    for (size_t k = 0; k < t.dimensions[2]; ++k) {
                        for (size_t l = 0; l < t.dimensions[3]; ++l) {
                            size_t idx = i * t.dimensions[1] * t.dimensions[2] * t.dimensions[3] +
                                         j * t.dimensions[2] * t.dimensions[3] +
                                         k * t.dimensions[3] + l;
                            os << t.data[idx];
                            if (l < t.dimensions[3] - 1) os << " ";
                        }
                        os << "\n";
                    }
                    os << "}";
                    if (j < t.dimensions[1] - 1) os << "\n";
                }
                os << "\n}";
                if (i < t.dimensions[0] - 1) os << "\n";
            }
            os << "\n}";
        } else if (N == 3) {
            os << "{\n";
            for (size_t i = 0; i < t.dimensions[0]; ++i) {
                os << "{\n";
                for (size_t j = 0; j < t.dimensions[1]; ++j) {
                    for (size_t k = 0; k < t.dimensions[2]; ++k) {
                        size_t idx = i * t.dimensions[1] * t.dimensions[2] +
                                     j * t.dimensions[2] + k;
                        os << t.data[idx];
                        if (k < t.dimensions[2] - 1) os << " ";
                    }
                    os << "\n";
                }
                os << "}";
                if (i < t.dimensions[0] - 1) os << "\n";
            }
            os << "\n}";
        } else if (N == 2) {
            os << "{\n";
            for (size_t i = 0; i < t.dimensions[0]; ++i) {
                for (size_t j = 0; j < t.dimensions[1]; ++j) {
                    os << t.data[i * t.dimensions[1] + j];
                    if (j < t.dimensions[1] - 1) os << " ";
                }
                os << "\n";
            }
            os << "}";
        } else if (N == 1) {
            for (size_t i = 0; i < t.dimensions[0]; ++i) {
                if (i) os << " ";
                os << t.data[i];
            }
            os << "\n";
        }
        return os;
    }

    template<typename U, size_t M>
    friend Tensor<U, M> transpose_2d(const Tensor<U, M>& tensor);

    template<typename U, size_t M>
    friend Tensor<U, M> matrix_product(const Tensor<U, M>& t1, const Tensor<U, M>& t2);
};

template<typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& tensor) {
    if (N < 2) {
        throw invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
    }

    auto new_dims = tensor.dimensions;
    swap(new_dims[N-2], new_dims[N-1]);

    Tensor<T, N> result(new_dims);

    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) batch_size *= tensor.dimensions[i];

    size_t rows = tensor.dimensions[N-2];
    size_t cols = tensor.dimensions[N-1];
    size_t matrix_size = rows * cols;

    for (size_t b = 0; b < batch_size; ++b) {
        size_t base_idx = b * matrix_size;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                size_t src_idx = base_idx + i * cols + j;
                size_t dst_idx = base_idx + j * rows + i;
                result.data[dst_idx] = tensor.data[src_idx];
            }
        }
    }
    return result;
}

template<typename T, size_t N>
Tensor<T, N> matrix_product(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
    if (N < 2) {
        throw invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    size_t m  = t1.dimensions[N-2];
    size_t k1 = t1.dimensions[N-1];
    size_t k2 = t2.dimensions[N-2];
    size_t n  = t2.dimensions[N-1];

    if (k1 != k2) {
        throw invalid_argument("Matrix dimensions are incompatible for multiplication");
    }

    for (size_t i = 0; i < N - 2; ++i) {
        if (t1.dimensions[i] != t2.dimensions[i]) {
            throw invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }
    }

    // (3) Multiplicación batcheada
    auto result_dims = t1.dimensions;
    result_dims[N-1] = n;
    Tensor<T, N> result(result_dims);

    size_t batch_size = 1;
    for (size_t i = 0; i < N - 2; ++i) batch_size *= t1.dimensions[i];

    for (size_t b = 0; b < batch_size; ++b) {
        size_t t1_base = b * m * k1;
        size_t t2_base = b * k2 * n;
        size_t res_base = b * m * n;

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T{};
                for (size_t p = 0; p < k1; ++p) {
                    sum += t1.data[t1_base + i * k1 + p] * t2.data[t2_base + p * n + j];
                }
                result.data[res_base + i * n + j] = sum;
            }
        }
    }
    return result;
}

} // namespace algebra
} // namespace utec

#endif
