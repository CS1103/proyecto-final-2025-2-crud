# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación de una red neuronal multicapa en C++ puro para clasificación y regresión, utilizando una arquitectura modular basada en tensores multidimensionales. El proyecto demuestra el uso de programación genérica avanzada, patrones de diseño y estructuras de datos eficientes para construir un framework de deep learning desde cero.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

## Datos Generales

* **Tema**: Redes Neuronales Artificiales - Implementación de Framework de Deep Learning en C++
* **Grupo**: `grupo_crud_neural_network`
* **Repositorio**: https://github.com/CS1103/proyecto-final-2025-2-crud.git
* **Integrantes**:

  * **Loli Gonzalez, Sebastian** – 202420022 (Responsable de Investigación Teórica y Documentación Académica)
  * **Aguirre Milla, Fernando** - 202420003 (Desarrollo de Arquitectura y Diseño de Patrones)
  * **Palomino Meza, Ricardo** - 202420152 (Implementación del Modelo y Estructuras Core)
  * **Gala Vásquez, Danna Nickol** – 202410573 (Análisis del Rendimiento 1)
  * **Choque Coaquira, Rafael** – 202410378 (Análisis del Rendimiento 2)

---

### Requisitos e instalación

#### **1. Compilador**
* **GCC 11** o superior (recomendado GCC 12+)
* **Clang 13** o superior (alternativa)
* Soporte completo para **C++17** o **C++20**

#### **2. Herramientas de Construcción**
* **CMake 3.18** o superior
* **Make** o **Ninja** (sistema de build)

#### **3. Dependencias**
* **Sistema operativo**: Linux (Ubuntu 20.04+), macOS (11+), o Windows (WSL2/MinGW)
* **Bibliotecas estándar de C++**: `<vector>`, `<array>`, `<algorithm>`, `<cmath>`, etc.
* **No se requieren dependencias externas** (implementación pura en C++)

### **Instalación Paso a Paso**

#### **En Linux/Ubuntu:**

```bash
# 1. Actualizar repositorios e instalar dependencias
sudo apt update
sudo apt install -y build-essential cmake git g++

# 2. Verificar versión de GCC (debe ser 11+)
g++ --version

# 3. Clonar el repositorio
git clone https://github.com/CS1103/proyecto-final-2025-2-crud.git
cd proyecto-final-2025-2-crud

# 4. Crear directorio de compilación
mkdir build && cd build

# 5. Configurar con CMake
cmake ..

# 6. Compilar el proyecto
make -j$(nproc)

# 7. Ejecutar pruebas (opcional)
./tests/neural_network_tests
```

#### **En macOS:**

```bash
# 1. Instalar Homebrew (si no está instalado)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Instalar herramientas de desarrollo
brew install gcc cmake git

# 3. Seguir pasos 3-7 de instalación en Linux
```

#### **En Windows (WSL2):**

```bash
# 1. Instalar WSL2 con Ubuntu 20.04+
wsl --install -d Ubuntu-22.04

# 2. Dentro de WSL, seguir pasos de instalación en Linux
```

### **Estructura del Proyecto**

```
proyecto-final-2025-2-crud/
├── include/                      # Archivos de cabecera (.h)
│   ├── tensor.h                 # Clase Tensor multidimensional
│   ├── neural_network.h         # Clase principal NeuralNetwork
│   ├── nn_interfaces.h          # Interfaces (ILayer, ILoss, IOptimizer)
│   ├── nn_dense.h               # Capa Dense (fully connected)
│   ├── nn_activation.h          # Funciones de activación (ReLU, Sigmoid)
│   ├── nn_loss.h                # Funciones de pérdida (MSE, BCE)
│   └── nn_optimizer.h           # Optimizadores (SGD, Adam)
├── src/                         # Implementaciones (.cpp)
│   └── main.cpp                 # Programa de ejemplo
├── tests/                       # Pruebas unitarias
│   ├── test_tensor.cpp
│   ├── test_layers.cpp
│   └── test_training.cpp
├── examples/                    # Ejemplos de uso
│   ├── xor_problem.cpp
│   ├── mnist_classifier.cpp
│   └── regression_demo.cpp
├── docs/                        # Documentación
│   ├── demo.mp4                # Video demostrativo
│   └── design_diagrams/        # Diagramas UML
├── CMakeLists.txt              # Configuración de CMake
├── README.md                   # Este archivo
└── LICENSE                     # Licencia MIT
```
---

### 1. Investigación teórica

### **Objetivo**

Explorar los fundamentos matemáticos y computacionales de las redes neuronales artificiales, comprender las arquitecturas principales, algoritmos de entrenamiento y técnicas de optimización, para implementar un framework funcional de deep learning en C++ desde cero.

---

### **1.1 Historia y Evolución de las Redes Neuronales**

#### **Orígenes (1943-1958)**

Las redes neuronales tienen sus raíces en el modelo McCulloch-Pitts (1943), que propuso la primera neurona artificial como una función lógica binaria. Este trabajo sentó las bases para el **Perceptrón** de Frank Rosenblatt (1958), el primer algoritmo de aprendizaje automático capaz de aprender patrones linealmente separables.

**Características del Perceptrón:**
- Función de activación escalón (step function)
- Aprendizaje mediante regla de actualización de pesos
- Limitación: solo puede resolver problemas linealmente separables

#### **Primer Invierno de IA (1969-1980s)**

Minsky y Papert (1969) demostraron matemáticamente que el perceptrón simple no podía resolver el problema XOR, limitando severamente su aplicabilidad. Esto llevó al primer "invierno de la inteligencia artificial" con reducción drástica de financiamiento.

#### **Renacimiento: Backpropagation (1986)**

El algoritmo de **retropropagación** (backpropagation) de Rumelhart, Hinton y Williams revolucionó el campo:
- Permitió entrenar redes con múltiples capas ocultas (Multilayer Perceptrons - MLP)
- Usa la regla de la cadena para calcular gradientes eficientemente
- Resolvió el problema XOR mediante capas ocultas no lineales

#### **Era Moderna (2006-Presente)**

Geoffrey Hinton introdujo el **Deep Learning** mediante:
- Preentrenamiento no supervisado (Restricted Boltzmann Machines)
- Redes profundas con múltiples capas ocultas
- GPUs para acelerar entrenamiento

**Hitos recientes:**
- 2012: AlexNet gana ImageNet (CNN profunda)
- 2014: GANs (Generative Adversarial Networks)
- 2017: Transformers y mecanismos de atención
- 2020+: Modelos masivos (GPT, BERT, etc.)

---

### **1.2 Arquitecturas Principales**

#### **1.2.1 Multilayer Perceptron (MLP)**

**Definición:** Red neuronal feedforward con al menos una capa oculta entre entrada y salida.

**Estructura:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Ecuaciones fundamentales:**

Para una capa con pesos W y bias b:
```
z = W·x + b
a = σ(z)
```

Donde:
- `x`: vector de entrada
- `W`: matriz de pesos (shape: [n_features, n_neurons])
- `b`: vector de bias
- `σ`: función de activación
- `z`: pre-activación
- `a`: activación (salida de la capa)

**Aplicaciones:**
- Clasificación de datos tabulares
- Regresión
- Problemas de aproximación de funciones

#### **1.2.2 Convolutional Neural Networks (CNN)**

**Definición:** Redes especializadas en procesar datos con estructura de grilla (imágenes, series temporales).

**Características clave:**
- **Convolución:** Operación de filtrado que detecta patrones locales
- **Pooling:** Reducción de dimensionalidad (max pooling, average pooling)
- **Compartición de pesos:** Los filtros se aplican a toda la imagen
- **Invarianza traslacional:** Detecta características independientemente de posición

**Ventajas:**
- Menos parámetros que MLP para imágenes
- Aprende jerarquías de características (bordes → texturas → objetos)
- Estado del arte en visión por computadora

**Aplicaciones:**
- Clasificación de imágenes
- Detección de objetos
- Segmentación semántica

#### **1.2.3 Recurrent Neural Networks (RNN)**

**Definición:** Redes con conexiones recurrentes que mantienen un estado interno (memoria).

**Variantes principales:**
- **Vanilla RNN:** Problema de gradientes que desaparecen
- **LSTM (Long Short-Term Memory):** Soluciona memoria a largo plazo
- **GRU (Gated Recurrent Unit):** Versión simplificada de LSTM

**Aplicaciones:**
- Procesamiento de lenguaje natural
- Series temporales
- Generación de secuencias

---

### **1.3 Algoritmos de Entrenamiento**

#### **1.3.1 Backpropagation (Retropropagación)**

**Concepto:** Algoritmo para calcular gradientes de la función de pérdida respecto a los pesos de la red, usando la regla de la cadena del cálculo diferencial.

**Proceso:**

1. **Forward Pass:** Calcular predicciones propagando datos hacia adelante
2. **Calcular pérdida:** L(ŷ, y)
3. **Backward Pass:** Propagar gradientes hacia atrás
4. **Actualizar pesos:** W ← W - η·∇W

**Matemática de Backpropagation:**

Para una red con capas L₁ → L₂ → ... → Ln:

```
δⁿ = ∇Loss                          # Gradiente en capa de salida
δⁿ⁻¹ = (Wⁿ)ᵀ · δⁿ ⊙ σ'(zⁿ⁻¹)       # Propagación hacia atrás
...
∇Wˡ = aˡ⁻¹ · (δˡ)ᵀ                  # Gradiente de pesos
```

Donde:
- `δˡ`: error en capa l
- `⊙`: producto elemento a elemento (Hadamard)
- `σ'`: derivada de función de activación

#### **1.3.2 Gradient Descent (Descenso de Gradiente)**

**Fórmula básica:**
```
θ(t+1) = θ(t) - η·∇J(θ)
```

**Variantes:**

1. **Batch Gradient Descent:** Usa todo el dataset en cada iteración
   - Ventaja: Convergencia estable
   - Desventaja: Lento para datasets grandes

2. **Stochastic Gradient Descent (SGD):** Actualiza con un ejemplo a la vez
   - Ventaja: Rápido, puede escapar mínimos locales
   - Desventaja: Convergencia ruidosa

3. **Mini-batch Gradient Descent:** Compromiso entre batch y SGD
   - Usa subconjuntos (típicamente 32, 64, 128, 256 ejemplos)
   - Balance entre velocidad y estabilidad

---

### **1.4 Optimizadores Avanzados**

#### **1.4.1 Stochastic Gradient Descent (SGD)**

**Implementación básica:**

```cpp
template
class SGD : public IOptimizer {
    T learning_rate;
    
    void update(Tensor& params, const Tensor& grads) override {
        // θ = θ - η·∇θ
        params -= learning_rate * grads;
    }
};
```

**Características:**
- Simple y computacionalmente eficiente
- Hiperparámetro único: learning rate (η)
- Requiere ajuste cuidadoso de η

**Variante - SGD con Momentum:**

```
v(t+1) = β·v(t) + ∇J(θ)
θ(t+1) = θ(t) - η·v(t+1)
```

- Acumula velocidad en dirección consistente
- Ayuda a superar mínimos locales
- β típicamente 0.9

#### **1.4.2 Adam (Adaptive Moment Estimation)**

**Algoritmo completo:**

```
m(t) = β₁·m(t-1) + (1-β₁)·∇θ          # Primer momento (media)
v(t) = β₂·v(t-1) + (1-β₂)·(∇θ)²       # Segundo momento (varianza)

m̂(t) = m(t) / (1 - β₁ᵗ)               # Corrección de sesgo
v̂(t) = v(t) / (1 - β₂ᵗ)               # Corrección de sesgo

θ(t+1) = θ(t) - η·m̂(t) / (√v̂(t) + ε)
```

**Implementación en este proyecto:**

```cpp
template
class Adam : public IOptimizer {
    T learning_rate, beta1, beta2, epsilon;
    int t;  // timestep
    unordered_map> m_cache;  // primer momento
    unordered_map> v_cache;  // segundo momento
    
    void update(Tensor& params, const Tensor& grads) override {
        // Actualizar momentos y parámetros
        // Ver implementación completa en nn_optimizer.h
    }
};
```

**Ventajas de Adam:**
- Tasa de aprendizaje adaptativa por parámetro
- Funciona bien con gradientes ruidosos
- Menos sensible a la elección de η
- Converge más rápido que SGD en muchos casos

**Hiperparámetros por defecto:**
- η = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

**Comparación SGD vs Adam:**

| Característica | SGD | Adam |
|---------------|-----|------|
| Velocidad | Más lento | Más rápido |
| Memoria | O(n) | O(3n) |
| Generalización | Mejor | Puede overfittear |
| Sensibilidad a η | Alta | Baja |
| Uso recomendado | Producción final | Prototipado rápido |

---

### **Conclusión de la Investigación Teórica**

Esta investigación proporciona la base teórica necesaria para:

1. **Comprender** cómo funcionan las redes neuronales desde fundamentos matemáticos
2. **Implementar** arquitecturas, optimizadores y funciones de pérdida de forma correcta
3. **Justificar** decisiones de diseño en el código (por qué ReLU, por qué Adam, etc.)
4. **Optimizar** el rendimiento basándose en propiedades matemáticas

El framework implementado en C++ materializa estos conceptos teóricos en estructuras de datos eficientes y algoritmos optimizados, demostrando la aplicabilidad práctica de la teoría de deep learning.

---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* PATRONES DE DISEÑO UTILIZADOS :
  
  Strategy Pattern (Patrón Estrategia)
  * ILayer, ILoss, IOptimizer
  * Propósito: Permite intercambiar algoritmos (capas, funciones de pérdida, optimizadores) sin modificar el cliente
  * Implementación: Interfaces abstractas con múltiples implementaciones concretas
 
  Template Method Pattern
  * Clase NeuralNetwork::train()
  * Propósito: Define el esqueleto del algoritmo de entrenamiento, delegando pasos específicos a las implementaciones
  
  Composite Pattern (Implícito)
  * std::vector<std::unique_ptr<ILayer<T>>>
  * Propósito: Trata capas individuales y composiciones de capas de manera uniforme
  
  Factory Method (Implícito)
  * Constructores de Dense con functores de inicialización
  * Propósito: Permite crear objetos con diferentes estrategias de inicialización
 
  Diagrama UML
```
┌─────────────────────────────┐
│      ILayer<T>              │
│  <<interface>>              │
├─────────────────────────────┤
│ + forward(): Tensor<T,2>    │
│ + backward(): Tensor<T,2>   │
│ + update_params()           │
└─────────────────────────────┘
         △                                  ┌─────────────────────────────┐
         │ implements                        │    ILoss<T, DIMS>           │
         ├──────────┬──────────┬──────────── │  <<interface>>              │  
         │          │          │             ├─────────────────────────────┤
    ┌────▼───┐  ┌───▼───┐  ┌──▼─────┐        │ + loss(): T                 │
    │ Dense  │  │ ReLU  │  │Sigmoid │        │ + loss_gradient(): Tensor   │
    └────────┘  └───────┘  └────────┘        └─────────────────────────────┘
                                                         △
                                                         │ implements
                                                         ├─────────────┬
                                                         │             │
                                                    ┌────▼─────┐  ┌────▼────┐
                                                    │ MSELoss  │  │ BCELoss │
                                                    └──────────┘  └─────────┘
 

┌─────────────────────────────┐
│    IOptimizer<T>            │
│  <<interface>>              │
├─────────────────────────────┤
│ + update()                  │
│ + step()                    │
└─────────────────────────────┘
         △
         │ implements
         ├───────────
         │           │
    ┌────▼───┐  ┌───▼────┐
    │  SGD   │  │  Adam  │
    └────────┘  └────────┘

┌─────────────────────────────────┐
│    NeuralNetwork<T>             │
├─────────────────────────────────┤
│ - layers: vector<ILayer<T>>     │
├─────────────────────────────────┤
│ + add_layer()                   │
│ + train()                       │
│ + predict()                     │
└─────────────────────────────────┘
         │ contains
         ▼
    ┌─────────┐
    │ ILayer  │
    └─────────┘

┌─────────────────────────────┐
│      Tensor<T, N>           │
├─────────────────────────────┤
│ - dimensions: array<N>      │
│ - data: vector<T>           │
├─────────────────────────────┤
│ + operator+,-,*,/           │
│ + reshape()                 │
│ + fill()                    │
│ + begin(), end()            │
└─────────────────────────────┘
```
* **Estructura de carpetas **:

```
proyecto/
│
├── include/
│   ├── tensor.h              # Clase Tensor base
│   ├── nn_interfaces.h       # Interfaces abstractas
│   ├── nn_dense.h           # Capa Dense (fully connected)
│   ├── nn_activation.h      # Funciones de activación
│   ├── nn_loss.h            # Funciones de pérdida
│   ├── nn_optimizer.h       # Optimizadores
│   └── neural_network.h     # Clase principal NeuralNetwork
│
├── src/
│   └── main.cpp             # Ejemplos de uso
│
├── tests/
│   ├── test_tensor.cpp      # Tests unitarios para Tensor
│   ├── test_layers.cpp      # Tests para capas
│   └── test_training.cpp    # Tests de entrenamiento
│
├── docs/
│   ├── manual_uso.md        # Este documento
│   └── uml_diagrams/        # Diagramas UML
│
├── CMakeLists.txt           # Configuración CMake
└── README.md                # Descripción general
```

#### 2.2 Manual de uso y casos de prueba

* **Creación de una Red Neuronal Básica**:
```cpp
#include "neural_network.h"
using namespace utec::neural_network;

// Crear red neuronal
NeuralNetwork<double> model;

// Agregar capas
model.add_layer(std::make_unique<Dense<double>>(
    2,      // 2 entradas
    4,      // 4 neuronas ocultas
    [](auto& w) { /* inicialización pesos */ },
    [](auto& b) { b.fill(0.0); }
));

model.add_layer(std::make_unique<ReLU<double>>());
model.add_layer(std::make_unique<Dense<double>>(4, 1, ...));
model.add_layer(std::make_unique<Sigmoid<double>>());
```
* **Entrenamiento**:
```cpp
// Datos de entrenamiento (Tensor 2D: [batch_size, features])
Tensor<double, 2> X(100, 2);  // 100 ejemplos, 2 features
Tensor<double, 2> Y(100, 1);  // 100 etiquetas

// Entrenar con MSE y SGD
model.train<MSELoss, SGD>(
    X,              // Datos de entrada
    Y,              // Etiquetas
    1000,           // Épocas
    32,             // Tamaño del batch (no implementado aún)
    0.01            // Learning rate
);
```
* **Predicción**:
```cpp
Tensor<double, 2> X_test(10, 2);  // 10 ejemplos de prueba
auto predictions = model.predict(X_test);
std::cout << predictions << std::endl;
```

* **Uso de Diferentes Optimizadores**:
```cpp
// Adam optimizer
model.train<BCELoss, Adam>(X, Y, 500, 32, 0.001);
```

* **Casos de prueba**:

Test 1: Tensor Operations

```cpp
TEST(TensorTest, BasicOperations) {
    Tensor<double, 2> t1(2, 3);
    t1.fill(1.0);
    
    Tensor<double, 2> t2(2, 3);
    t2.fill(2.0);
    
    auto result = t1 + t2;
    EXPECT_EQ(result(0,0), 3.0);
}
```
Test 2: Dense Forward Pass

```cpp
TEST(DenseTest, ForwardPass) {
    Dense<double> layer(2, 3,
        [](auto& w) { w.fill(0.5); },
        [](auto& b) { b.fill(0.1); }
    );
    
    Tensor<double, 2> input(1, 2);
    input = {1.0, 2.0};
    
    auto output = layer.forward(input);
    // output debería ser (1*0.5 + 2*0.5 + 0.1) para cada neurona
    EXPECT_NEAR(output(0,0), 1.6, 1e-6);
}
```

Test 3: Backpropagation

```cpp
TEST(DenseTest, Backward) {
    Dense<double> layer(2, 1, ...);
    
    Tensor<double, 2> input(1, 2);
    input = {1.0, 2.0};
    
    auto output = layer.forward(input);
    
    Tensor<double, 2> grad_output(1, 1);
    grad_output = {1.0};
    
    auto grad_input = layer.backward(grad_output);
    EXPECT_EQ(grad_input.shape()[0], 1);
    EXPECT_EQ(grad_input.shape()[1], 2);
}
```

Test 4: Loss Functions

```cpp
TEST(LossTest, MSELoss) {
    Tensor<double, 2> pred(2, 1);
    pred = {0.8, 0.2};
    
    Tensor<double, 2> target(2, 1);
    target = {1.0, 0.0};
    
    MSELoss<double> loss(pred, target);
    double loss_value = loss.loss();
    
    // MSE = ((0.8-1)^2 + (0.2-0)^2) / 2 = 0.04/2 = 0.02
    EXPECT_NEAR(loss_value, 0.04, 1e-6);
}
```

Test 5: Activation Functions

```cpp
TEST(ActivationTest, ReLU) {
    ReLU<double> relu;
    
    Tensor<double, 2> input(2, 2);
    input = {{-1.0, 2.0}, {-3.0, 4.0}};
    
    auto output = relu.forward(input);
    EXPECT_EQ(output(0,0), 0.0);
    EXPECT_EQ(output(0,1), 2.0);
    EXPECT_EQ(output(1,0), 0.0);
    EXPECT_EQ(output(1,1), 4.0);
}
```
#### 2.3  Ejemplos de uso con código


Ejemplo 1: XOR Problem

```cpp
#include "neural_network.h"
using namespace utec::neural_network;

int main() {
    // Datos XOR
    Tensor<double, 2> X(4, 2);
    X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    
    Tensor<double, 2> Y(4, 1);
    Y = {0, 1, 1, 0};
    
    // Crear modelo
    NeuralNetwork<double> model;
    
    // Arquitectura: 2 -> 4 -> 1
    model.add_layer(std::make_unique<Dense<double>>(
        2, 4,
        [](auto& w) { 
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> dist(0.0, 0.5);
            for(auto& val : w) val = dist(gen);
        },
        [](auto& b) { b.fill(0.0); }
    ));
    
    model.add_layer(std::make_unique<ReLU<double>>());
    
    model.add_layer(std::make_unique<Dense<double>>(
        4, 1,
        [](auto& w) { /* inicialización */ },
        [](auto& b) { b.fill(0.0); }
    ));
    
    model.add_layer(std::make_unique<Sigmoid<double>>());
    
    // Entrenar
    model.train<MSELoss, Adam>(X, Y, 5000, 4, 0.01);
    
    // Predecir
    auto predictions = model.predict(X);
    std::cout << "Predicciones:\n" << predictions;
    
    return 0;
}
```

Ejemplo 2: Clasificación Binaria

```cpp
// Generar datos sintéticos
Tensor<double, 2> X(100, 2);
Tensor<double, 2> Y(100, 1);

// Clase 0: puntos alrededor de (0, 0)
// Clase 1: puntos alrededor de (1, 1)
std::random_device rd;
std::mt19937 gen(rd());
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
model.add_layer(std::make_unique<Dense<double>>(2, 8, ...));
model.add_layer(std::make_unique<ReLU<double>>());
model.add_layer(std::make_unique<Dense<double>>(8, 1, ...));
model.add_layer(std::make_unique<Sigmoid<double>>());

model.train<BCELoss, SGD>(X, Y, 1000, 32, 0.1);
```

Ejemplo 3: Regresión

```cpp
// Datos de regresión: y = 2x + 1 + ruido
Tensor<double, 2> X(100, 1);
Tensor<double, 2> Y(100, 1);

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> x_dist(0.0, 10.0);
std::normal_distribution<> noise(0.0, 0.5);

for(size_t i = 0; i < 100; ++i) {
    double x = x_dist(gen);
    X(i, 0) = x;
    Y(i, 0) = 2.0 * x + 1.0 + noise(gen);
}

// Modelo simple: 1 -> 1 (sin activación)
NeuralNetwork<double> model;
model.add_layer(std::make_unique<Dense<double>>(1, 1, ...));

model.train<MSELoss, SGD>(X, Y, 1000, 32, 0.01);

// Predecir para nuevos valores
Tensor<double, 2> X_test(5, 1);
X_test = {1.0, 2.0, 3.0, 4.0, 5.0};
auto predictions = model.predict(X_test);
```


---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Loli Gonzalez, Sebastian | Documentar bases teóricas |
| Diseño de la arquitectura | Aguirre Milla, Fernando | UML y esquemas de clases  |
| Implementación del modelo | Palomino Meza, Ricardo | Código C++ de la NN       |
| Análisis del Rendimiento 1    | Gala Vásquez, Danna Nickol | Generación de métricas    |
| Análisis del Rendimiento 2      | Choque Coaquira, Rafael | Comparación de optimizadores     |

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

[1] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016.

[2] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533-536, 1986.

[3] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. 3rd Int. Conf. Learn. Representations (ICLR)*, 2015.

[4] K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2015, pp. 1026-1034.

---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
