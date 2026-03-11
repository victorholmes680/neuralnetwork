# nn.h - Neural Network Library Explained

## Table of Contents
1. [Overview](#overview)
2. [Interactive Learning Guide](#interactive-learning-guide) ⭐ **START HERE**
3. [Header Guards](#header-guards)
4. [Includes and Dependencies](#includes-and-dependencies)
5. [Configurable Macros](#configurable-macros)
6. [Data Structures](#data-structures)
7. [Matrix Operations](#matrix-operations)
8. [Neural Network Structure](#neural-network-structure)
9. [Training Algorithm](#training-algorithm)
10. [Visualization System (Gym)](#visualization-system-gym)

---

## Overview

`nn.h` is a **header-only C library** that implements a simple feedforward neural network with backpropagation training. It's designed to be educational and easy to understand.

### Key Features
- **Matrix operations** for mathematical computations
- **Multi-layer perceptron** neural network
- **Backpropagation** training algorithm
- **Optional visualization** using Raylib library

---

## Interactive Learning Guide

> 🎯 **This section is designed to help you rewrite nn.h from scratch!**
> Follow these guided exercises to deeply understand how the library works.

### Learning Path

Choose your path based on your experience level:

| Level | Description | Where to Start |
|-------|-------------|----------------|
| 🟢 **Beginner** | New to C and neural networks? Start here | [Step 1: Understanding the Matrix](#step-1-understanding-the-matrix) |
| 🟡 **Intermediate** | Know C but new to NNs? | [Step 3: Matrix Operations](#step-3-matrix-operations) |
| 🔴 **Advanced** | Understand the concepts, want practice? | [Step 5: Neural Network](#step-5-neural-network) |

---

### Step 1: Understanding the Matrix
#### 🤔 Think About It

Before writing any code, answer these questions:

**Q1: What is a matrix?**
<details>
<summary>Click to see answer</summary>

A matrix is a 2-dimensional grid of numbers. In C, we represent it as:
- Number of rows
- Number of columns
- A pointer to the actual data
</details>

**Q2: How do we store a 2D array in C memory?**
<details>
<summary>Click to see answer</summary>

We store it in **row-major order** - all elements of row 0 first, then row 1, etc.

Example for a 2×3 matrix:
```
[1, 2, 3]     Memory: [1, 2, 3, 4, 5, 6]
[4, 5, 6]
```
</details>

**Q3: How do we access element at row i, column j?**
<details>
<summary>Click to see answer</summary>

```
index = i * columns + j
```

For a 3×4 matrix, element [1][2] is at index:
```
index = 1 * 4 + 2 = 6
```
</details>

#### ✍️ Your Turn: Write the Matrix Structure

Now try to write the structure yourself before looking at the answer:

```c
// TODO: Write the Mat structure
// It should have:
// - rows (number of rows)
// - cols (number of columns)
// - stride (why do we need this?)
// - es (pointer to elements)

typedef struct {
    // Your code here
} Mat;
```

<details>
<summary>Click to see the answer</summary>

```c
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of columns
    size_t stride;  // Elements per row in memory
    float *es;      // Pointer to matrix elements
} Mat;
```
</details>

#### 🤔 What is `stride` and why do we need it?

<details>
<summary>Click for explanation</summary>

Stride is the number of elements to skip to get to the next row. For most matrices, `stride == cols`.

**But stride enables a powerful feature:** row views that share memory!

```c
// Parent matrix: 3×4
Mat parent = {
    .rows = 3,
    .cols = 4,
    .stride = 4,
    .es = [1,2,3,4, 5,6,7,8, 9,10,11,12]
};

// Row view: 1×4 (shares memory with parent!)
Mat row = {
    .rows = 1,
    .cols = 4,
    .stride = 4,  // Same as parent!
    .es = &parent.es[4]  // Points to row 1
};

// Modifying row modifies parent!
MAT_AT(row, 0, 0) = 99;  // Changes parent[1][0] to 99
```

This avoids expensive memory copies!
</details>

---

### Step 2: Matrix Access Macro

#### 🤔 Think About It

Given what you learned about stride, how do we access element [i][j]?

**Hint:** Remember the memory is laid out as rows, and stride tells us how far apart rows are.

<details>
<summary>Click to see answer</summary>

```c
#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
```

**Breakdown:**
- `(i) * (m).stride` - skip i rows
- `+ (j)` - move j columns into the current row
</details>

#### ✍️ Practice: Trace Through an Example

```c
Mat m = {
    .rows = 2,
    .cols = 3,
    .stride = 3,
    .es = {1, 2, 3, 4, 5, 6}
};
```

**What is MAT_AT(m, 1, 2)?**

<details>
<summary>Click to see answer</summary>

```c
MAT_AT(m, 1, 2) = m.es[1 * 3 + 2] = m.es[5] = 6
```
</details>

**What if stride = 4 (larger than cols)?**

<details>
<summary>Click to see answer</summary>

```c
MAT_AT(m, 1, 2) = m.es[1 * 4 + 2] = m.es[6]

// The extra element in stride is "padding" or unused
```
</details>

---

### Step 3: Matrix Operations

#### ✍️ Exercise: Implement mat_alloc()

Before looking at the answer, try to write `mat_alloc()`:

```c
// Allocate a matrix with given dimensions
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    // Your code here:
    // 1. Set the dimensions
    // 2. Allocate memory
    // 3. Return the matrix

    return m;
}
```

<details>
<summary>Click to see the answer</summary>

```c
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;  // For new matrices, stride = cols
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL && "Memory allocation failed");
    return m;
}
```
</details>

#### ✍️ Exercise: Implement mat_fill()

Try to write a function that fills every element with a value:

```c
void mat_fill(Mat m, float x)
{
    // Your code here: iterate through all elements
}
```

<details>
<summary>Click to see the answer</summary>

```c
void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}
```
</details>

#### 🎯 Challenge: Matrix Multiplication

This is the hardest but most important operation!

**Understanding the Problem:**

If A is 2×3 and B is 3×4, what size is C = A × B?

<details>
<summary>Click to see answer</summary>

C is 2×4

**Rule:** (m×n) × (n×p) = (m×p)

The inner dimensions must match!
</details>

**How to compute C[i][j]?**

<details>
<summary>Click to see answer</summary>

```
C[i][j] = Σ(A[i][k] * B[k][j]) for k = 0 to n-1
```

This is the dot product of row i of A and column j of B.

**Example:**
```
A = [1 2]    B = [5 6]
    [3 4]        [7 8]

C[0][0] = 1*5 + 2*7 = 19
C[0][1] = 1*6 + 2*8 = 22
C[1][0] = 3*5 + 4*7 = 43
C[1][1] = 3*6 + 4*8 = 50
```
</details>

#### ✍️ Exercise: Implement mat_dot()

```c
void mat_dot(Mat dst, Mat a, Mat b)
{
    // TODO: Implement matrix multiplication
    // Remember to:
    // 1. Assert dimensions are compatible
    // 2. Initialize dst to zero
    // 3. Compute each element
}
```

<details>
<summary>Click to see the answer</summary>

```c
void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;  // Initialize
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}
```
</details>

---

### Step 4: Activation Functions

#### 🤔 What is an activation function?

<details>
<summary>Click for explanation</summary>

Activation functions introduce non-linearity into neural networks. Without them, a neural network is just matrix multiplication, which can only learn linear relationships.

**Common activation functions:**
- **Sigmoid:** Squashes to (0, 1) - good for probabilities
- **ReLU:** max(0, x) - simple and fast
- **Tanh:** Squashes to (-1, 1) - zero-centered
</details>

#### The Sigmoid Function

```
σ(x) = 1 / (1 + e^(-x))
```

**Properties:**
- Always outputs between 0 and 1
- Smooth and differentiable
- Derivative is easy: σ'(x) = σ(x) × (1 - σ(x))

#### ✍️ Exercise: Implement sigmoidf()

```c
float sigmoidf(float x)
{
    // TODO: Implement sigmoid
    // Hint: Use expf() from math.h
}
```

<details>
<summary>Click to see the answer</summary>

```c
float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
```
</details>

#### ✍️ Exercise: Apply sigmoid to a matrix

```c
void mat_sig(Mat m)
{
    // TODO: Apply sigmoid to every element
}
```

<details>
<summary>Click to see the answer</summary>

```c
void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}
```
</details>

---

### Step 5: Neural Network

#### 🤔 What is a neural network?

<details>
<summary>Click for explanation</summary>

A neural network is composed of layers of neurons:
- **Input layer:** Receives the raw data
- **Hidden layers:** Process the data (can have multiple)
- **Output layer:** Produces the final prediction

Each connection between neurons has:
- **Weight (W):** How strong the connection is
- **Bias (b):** An offset/threshold

**Forward pass:**
```
output = σ(input × W + b)
```
</details>

#### Understanding the NN Structure

For architecture `[2, 3, 1]`:

```
Layer 0 (Input): 2 neurons
Layer 1 (Hidden): 3 neurons
Layer 2 (Output): 1 neuron
```

**What matrices do we need?**

<details>
<summary>Click to see answer</summary>

```c
typedef struct {
    size_t count;  // Number of layers with parameters (2)
    Mat *ws;       // Weight matrices
    Mat *bs;       // Bias matrices
    Mat *as;       // Activation matrices
} NN;
```

For `[2, 3, 1]`:
- `count = 2`
- `ws[0]`: 2×3 (input → hidden)
- `bs[0]`: 1×3 (hidden biases)
- `ws[1]`: 3×1 (hidden → output)
- `bs[1]`: 1×1 (output bias)
- `as[0]`: 1×2 (input activations)
- `as[1]`: 1×3 (hidden activations)
- `as[2]`: 1×1 (output activations)
</details>

#### ✍️ Exercise: Implement the Forward Pass

The forward pass computes the network's output:

```c
void nn_forward(NN nn)
{
    // TODO: For each layer:
    // 1. Multiply by weights: a × W
    // 2. Add bias: y + b
    // 3. Apply activation: σ(y)
}
```

<details>
<summary>Click to see the answer</summary>

```c
void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);  // a × W
        mat_sum(nn.as[i + 1], nn.bs[i]);             // + b
        mat_sig(nn.as[i + 1]);                       // σ()
    }
}
```
</details>

---

### Step 6: Training - The Cost Function

#### 🤔 How do we know if the network is good?

<details>
<summary>Click for explanation</summary>

We use a **cost function** (also called loss function) that measures the error.

**Mean Squared Error (MSE):**
```
J = (1/n) × Σ(predicted - actual)²
```

Lower cost = better predictions!
</details>

#### ✍️ Exercise: Implement nn_cost()

```c
float nn_cost(NN nn, Mat ti, Mat to)
{
    // TODO:
    // 1. For each training sample
    // 2. Run forward pass
    // 3. Calculate squared error
    // 4. Return mean error

    return 0.0f;  // Placeholder
}
```

<details>
<summary>Click to see the answer</summary>

```c
float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;
    float c = 0;

    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        for (size_t j = 0; j < to.cols; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d * d;
        }
    }
    return c / n;
}
```
</details>

---

### Step 7: Backpropagation

#### 🤔 How do we improve the network?

<details>
<summary>Click for explanation</summary>

We use **gradient descent**:
1. Calculate how much each parameter contributes to the error (gradient)
2. Adjust parameters in the opposite direction of the gradient
3. Repeat!

**The Chain Rule:**
```
∂Loss/∂Weight = ∂Loss/∂Output × ∂Output/∂Activation × ∂Activation/∂Weight
```

Backpropagation efficiently computes all gradients using the chain rule!
</details>

#### Key Gradients for Sigmoid:

For a single neuron with:
- `a` = activation value
- `da` = error gradient

```
Bias gradient:  ∂J/∂b = 2 × da × a × (1 - a)
Weight gradient: ∂J/∂W = 2 × da × a × (1 - a) × previous_activation
```

#### ✍️ Exercise: Implement Gradient Descent Update

```c
void nn_learn(NN nn, NN g, float rate)
{
    // TODO: Update each parameter:
    // parameter = parameter - learning_rate × gradient
}
```

<details>
<summary>Click to see the answer</summary>

```c
void nn_learn(NN nn, NN g, float rate)
{
    for (size_t i = 0; i < nn.count; ++i) {
        // Update weights
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }
        // Update biases
        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}
```
</details>

---

### Step 8: Putting It All Together

#### ✍️ Complete Exercise: Train XOR

Try to write a complete program that trains a neural network to solve XOR:

**XOR Truth Table:**
```
Input | Output
------|-------
0, 0  | 0
0, 1  | 1
1, 0  | 1
1, 1  | 0
```

<details>
<summary>Click for a complete solution</summary>

```c
#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>

int main(void)
{
    // Define architecture: 2 inputs, 3 hidden, 1 output
    size_t arch[] = {2, 3, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));

    nn_rand(nn, -1, 1);

    // Training data
    float train_data[] = {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0
    };

    Mat ti = {.rows = 4, .cols = 2, .stride = 3, .es = train_data};
    Mat to = {.rows = 4, .cols = 1, .stride = 3, .es = train_data + 2};

    // Training loop
    float rate = 0.1f;
    for (size_t i = 0; i < 10000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);

        if (i % 1000 == 0) {
            printf("Cost: %f\n", nn_cost(nn, ti, to));
        }
    }

    // Test
    mat_fill(NN_INPUT(nn), 0);
    MAT_AT(NN_INPUT(nn), 0, 0) = 1;
    MAT_AT(NN_INPUT(nn), 0, 1) = 0;
    nn_forward(nn);
    printf("XOR(1, 0) = %f\n", MAT_AT(NN_OUTPUT(nn), 0, 0));

    return 0;
}
```
</details>

---

### 🎓 Congratulations!

You've now learned:
- ✅ How to represent matrices in C
- ✅ Matrix operations (multiplication, etc.)
- ✅ Activation functions
- ✅ Forward pass in neural networks
- ✅ Cost functions
- ✅ Gradient descent
- ✅ Training a neural network

**Next Steps:**
- Experiment with different architectures
- Try different activation functions (ReLU, Tanh)
- Learn about optimization techniques (momentum, Adam)
- Study regularization (dropout, L2)

---

### 📚 Quick Reference

**Essential Concepts:**

| Concept | Formula | Purpose |
|---------|---------|---------|
| Matrix Access | `index = i * stride + j` | Find element in memory |
| Matrix Multiply | `C[i][j] = Σ(A[i][k] * B[k][j])` | Transform data |
| Sigmoid | `σ(x) = 1/(1 + e^(-x))` | Squash to (0,1) |
| Forward Pass | `a = σ(input × W + b)` | Compute prediction |
| MSE Cost | `J = Σ(predicted - actual)² / n` | Measure error |
| Gradient Descent | `param = param - rate × gradient` | Minimize error |

---

## Header Guards

```c
#ifndef NN_H_
#define NN_H_
...
#endif // NN_H_
```

### What are header guards?
Header guards prevent the same header file from being included multiple times in a single compilation unit. Without them, you'd get "redefinition" errors.

### How they work:
1. First time: `NN_H_` is not defined, so it gets defined and content is included
2. Subsequent times: `NN_H_` is already defined, so the entire file is skipped

---

## Includes and Dependencies

```c
#include <stdbool.h>   // For bool type
#include <stddef.h>    // For size_t type
#include <stdio.h>     // For file I/O and printf
#include <time.h>      // For random number seeding
#include <math.h>      // For expf() and sigmoid function
#include <stdint.h>    // For fixed-width integers
#include <float.h>     // For float constants (FLT_MAX, etc.)
#include <string.h>    // For memcpy, strlen
```

Each include provides specific functionality:
- **stdbool.h**: Boolean type (`true`, `false`)
- **stddef.h**: Size type and NULL constant
- **stdio.h**: Input/output operations
- **math.h**: Mathematical functions
- **stdlib.h**: Memory allocation (malloc)

---

## Configurable Macros

### Memory Allocation Customization

```c
#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC
```

### Why this is useful:
- Allows users to inject custom memory allocators
- Useful for debugging (can track allocations)
- Useful for embedded systems (might use different allocation strategy)

### Assertion Customization

```c
#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT
```

Similar to `NN_MALLOC`, this allows custom assertion behavior.

### Array Length Macro

```c
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
```

This calculates the number of elements in a static array.

**Example:**
```c
int arr[] = {1, 2, 3, 4, 5};
size_t len = ARRAY_LEN(arr); // len = 5
```

---

## Data Structures

### 1. Gym_Batch - Batch Processing State

```c
typedef struct {
    size_t begin;      // Starting index for current batch
    float cost;        // Accumulated cost for the batch
    bool finished;     // Whether batch processing is complete
} Gym_Batch;
```

This tracks progress when processing data in mini-batches for training.

### 2. Mat - Matrix Structure

```c
typedef struct {
    size_t rows;      // Number of rows
    size_t cols;      // Number of columns
    size_t stride;    // Elements per row in memory (enables row views)
    float *es;        // Pointer to matrix elements
} Mat;
```

#### Understanding `stride`:
The stride is an advanced feature that allows efficient row operations. A normal matrix has `stride == cols`, but row views can share memory with the parent matrix.

**Example:**
```c
// Original matrix: 3x4
// Memory layout: [a,b,c,d, e,f,g,h, i,j,k,l]

// Row 1 view: 1x4
// stride = 4 (same as parent)
// es points to 'e' element
```

### Matrix Access Macro

```c
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]
```

This macro accesses element at row `i`, column `j`.

**How it works:**
```
Memory index = row * stride + column
```

**Example:**
```c
Mat m = {
    .rows = 2,
    .cols = 3,
    .stride = 3,
    .es = {1,2,3, 4,5,6}
};

MAT_AT(m, 1, 2) = m.es[1*3 + 2] = m.es[5] = 6
```

---

## Matrix Operations

### 1. mat_alloc() - Create a Matrix

```c
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;  // stride equals cols for new matrices
    m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
    NN_ASSERT(m.es != NULL);
    return m;
}
```

### What it does:
1. Sets matrix dimensions
2. Allocates memory for all elements
3. Returns the initialized matrix

### 2. mat_fill() - Fill Matrix with Value

```c
void mat_fill(Mat m, float x)
{
    for(size_t i = 0; i < m.rows; ++i) {
        for(size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}
```

### 3. mat_dot() - Matrix Multiplication

```c
void mat_dot(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for(size_t i = 0; i < dst.rows; ++i) {
        for(size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;
            for(size_t k = 0; k < a.cols; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}
```

### Understanding Matrix Multiplication:

Given matrices A (m×n) and B (n×p), the product C = A×B (m×p) is:

```
C[i][j] = Σ(A[i][k] * B[k][j]) for k = 0 to n-1
```

**Example:**
```
A = [1 2]    B = [5 6]
    [3 4]        [7 8]

C[0][0] = 1*5 + 2*7 = 5 + 14 = 19
C[0][1] = 1*6 + 2*8 = 6 + 16 = 22
C[1][0] = 3*5 + 4*7 = 15 + 28 = 43
C[1][1] = 3*6 + 4*8 = 18 + 32 = 50

C = [19 22]
    [43 50]
```

### 4. mat_row() - Create a Row View

```c
Mat mat_row(Mat m, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,  // Share stride with parent
        .es = &MAT_AT(m, row, 0),  // Point to row's first element
    };
}
```

This creates a 1×cols matrix that **shares memory** with the parent matrix. No copying occurs!

### 5. mat_sig() - Apply Sigmoid Activation

```c
void mat_sig(Mat m)
{
    for(size_t i = 0; i < m.rows; ++i) {
        for(size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}
```

### The Sigmoid Function:
```
σ(x) = 1 / (1 + e^(-x))
```

**Properties:**
- Output range: (0, 1)
- Squashes any input to between 0 and 1
- Differentiable (important for backpropagation)

**Graph:**
```
     1 |            ________
       |          /
       |        /
       |      /
       |    /
     0 |__/
       +------------------
        -∞       0       +∞
```

### 6. mat_sum() - Element-wise Addition

```c
void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for(size_t i = 0; i < dst.rows; ++i) {
        for(size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}
```

Adds corresponding elements: `dst[i][j] += a[i][j]`

### 7. mat_copy() - Copy Matrix Data

```c
void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);
    for(size_t i = 0; i < dst.rows; ++i) {
        for(size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}
```

### 8. mat_rand() - Random Initialization

```c
void mat_rand(Mat m, float low, float high)
{
    for(size_t i = 0; i < m.rows; ++i) {
        for(size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

float rand_float(void)
{
    return (float)rand()/(float)RAND_MAX;
}
```

Generates random floats in range [low, high].

### 9. mat_save() and mat_load() - Persistence

```c
void mat_save(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);
    for(size_t i = 0; i < m.rows; ++i) {
        size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, out);
        while(n < m.cols && !ferror(out)) {
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
            n += k;
        }
    }
}
```

Saves matrix to file with magic number for validation.

### 10. mat_print() - Debug Output

```c
void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n",(int)padding, "", name);
    for(size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int)padding, "");
        for(size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

#define MAT_PRINT(m) mat_print(m, #m, 0)
```

The `MAT_PRINT` macro uses stringification (`#m`) to automatically use the variable name as the label.

### 11. mat_shuffle_rows() - Random Row Shuffling

```c
void mat_shuffle_rows(Mat m)
{
    for(size_t i = 0; i < m.rows; ++i) {
        size_t j = i + rand() % (m.rows - i);
        if(i != j) {
            for(size_t k = 0; k < m.cols; ++k) {
                float t = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = t;
            }
        }
    }
}
```

Uses Fisher-Yates shuffle algorithm for randomness. Important for training to avoid order bias.

---

## Neural Network Structure

### The NN Structure

```c
typedef struct {
    size_t count;    // Number of layers (input layer not counted)
    Mat *ws;         // Array of weight matrices
    Mat *bs;         // Array of bias matrices
    Mat *as;         // Array of activation matrices
} NN;
```

### Understanding the Structure:

For a network with architecture `[2, 3, 1]` (2 inputs, 3 hidden, 1 output):
- `count = 2` (two layers with learnable parameters)
- `ws[0]` = 2×3 matrix (input → hidden weights)
- `bs[0]` = 1×3 matrix (hidden layer biases)
- `ws[1]` = 3×1 matrix (hidden → output weights)
- `bs[1]` = 1×1 matrix (output bias)
- `as[0]` = 1×2 matrix (input layer)
- `as[1]` = 1×3 matrix (hidden activations)
- `as[2]` = 1×1 matrix (output activations)

### Helper Macros

```c
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
```

These provide convenient access to input and output layers.

### nn_alloc() - Create Network

```c
NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);
    NN nn;
    nn.count = arch_count - 1;  // Number of weight layers

    nn.ws = NN_MALLOC(sizeof(*nn.ws)*nn.count);
    nn.bs = NN_MALLOC(sizeof(*nn.bs)*nn.count);
    nn.as = NN_MALLOC(sizeof(*nn.as)*(nn.count + 1));

    nn.as[0] = mat_alloc(1, arch[0]);  // Input layer

    for(size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = mat_alloc(nn.as[i-1].rows, arch[i]);
        nn.as[i]   = mat_alloc(nn.as[i-1].rows, arch[i]);
    }

    return nn;
}
```

### How it works:
1. Calculate number of layers with parameters
2. Allocate arrays for weights, biases, activations
3. Create each layer with appropriate dimensions

### nn_forward() - Forward Pass

```c
void nn_forward(NN nn)
{
    for(size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);  // Multiply by weights
        mat_sum(nn.as[i+1], nn.bs[i]);             // Add bias
        mat_sig(nn.as[i+1]);                       // Apply activation
    }
}
```

### The Forward Pass:

For each layer:
1. **Weighted sum**: `y = a_prev × W`
2. **Add bias**: `y = y + b`
3. **Activation**: `a = σ(y)`

**Mathematical representation:**
```
a^(l) = σ(a^(l-1) × W^(l) + b^(l))
```

### nn_cost() - Calculate Loss

```c
float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols);

    size_t n = ti.rows;
    float c = 0;

    for(size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);  // Input sample
        Mat y = mat_row(to, i);  // Target output

        mat_copy(NN_INPUT(nn), x);
        nn_forward(nn);

        size_t q = to.cols;
        for(size_t j = 0; j < q; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d*d;  // Squared error
        }
    }
    return c / n;  // Mean squared error
}
```

### Understanding Cost Function:

**Mean Squared Error (MSE):**
```
J = (1/n) × Σ(predicted - actual)²
```

This measures how wrong the network's predictions are. Lower is better!

---

## Training Algorithm

### nn_backprop() - Backpropagation

```c
void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows);
    size_t n = ti.rows;
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);  // Reset gradients

    for(size_t i = 0; i < n; ++i) {
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        // Reset activation gradients
        for(size_t j = 0; j <= nn.count; ++j) {
            mat_fill(g.as[j], 0);
        }

        // Calculate output layer error
        for(size_t j = 0; j < to.cols; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        // Backpropagate through layers
        for(size_t l = nn.count; l > 0; --l) {
            for(size_t j = 0; j < nn.as[l].cols; ++j) {
                float a  = MAT_AT(nn.as[l], 0, j);      // Activation
                float da = MAT_AT(g.as[l], 0, j);       // Gradient

                // Bias gradient: 2*da*a*(1-a)
                MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1-a);

                for(size_t k = 0; k < nn.as[l-1].cols; ++k) {
                    float pa = MAT_AT(nn.as[l-1], 0, k);  // Previous activation
                    float w  = MAT_AT(nn.ws[l-1], k, j);   // Weight

                    // Weight gradient
                    MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1-a)*pa;

                    // Backpropagate to previous layer
                    MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1-a)*w;
                }
            }
        }
    }

    // Average gradients over all samples
    for(size_t i = 0; i < g.count; ++i) {
        for(size_t j = 0; j < g.ws[i].rows; ++j) {
            for(size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for(size_t j = 0; j < g.bs[i].rows; ++j) {
            for(size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}
```

### Understanding Backpropagation:

Backpropagation calculates how much each parameter contributes to the error.

**The Chain Rule:**
```
∂Loss/∂Weight = ∂Loss/∂Output × ∂Output/∂Activation × ∂Activation/∂Weight
```

**For sigmoid activation:**
```
σ'(x) = σ(x) × (1 - σ(x))
```

**Gradient equations:**
```
∂J/∂b = 2 × (predicted - actual) × a × (1 - a)
∂J/∂W = 2 × (predicted - actual) × a × (1 - a) × previous_activation
```

### nn_learn() - Update Weights

```c
void nn_learn(NN nn, NN g, float rate)
{
    for(size_t i = 0; i < nn.count; ++i) {
        for(size_t j = 0; j < nn.ws[i].rows; ++j) {
            for(size_t k = 0; k < nn.ws[i].cols; ++k) {
                MAT_AT(nn.ws[i], j, k) -= rate * MAT_AT(g.ws[i], j, k);
            }
        }

        for(size_t j = 0; j < nn.bs[i].rows; ++j) {
            for(size_t k = 0; k < nn.bs[i].cols; ++k) {
                MAT_AT(nn.bs[i], j, k) -= rate * MAT_AT(g.bs[i], j, k);
            }
        }
    }
}
```

### Gradient Descent:

```
parameter = parameter - learning_rate × gradient
```

This moves parameters in the direction that reduces the error.

**Visual analogy:**
```
Imagine you're on a mountain (the error surface)
- You want to go down (minimize error)
- Gradient tells you which way is up
- You move opposite to gradient (go down)
- Learning rate = step size
```

### nn_rand() - Random Initialization

```c
void nn_rand(NN nn, float low, float high)
{
    for(size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}
```

Initializes all weights and biases with random values.

---

## Visualization System (Gym)

The gym system (enabled with `#define NN_ENABLE_GYM`) provides visualization using Raylib.

### Plot Structure

```c
typedef struct {
    float *items;      // Array of values to plot
    size_t count;      // Number of values
    size_t capacity;   // Allocated capacity
} Plot;
```

### Dynamic Array Macro

```c
#define da_append(da, item)                         \
do {                                                \
    if(((da))->count >= ((da))->capacity) {         \
        (da)->capacity = (da)->capacity == 0 ?      \
            DA_INIT_CAP : (da)->capacity*2;         \
        (da)->items = realloc((da)->items,          \
            (da)->capacity*sizeof(*(da)->items));   \
        assert((da)->items != NULL);                \
    }                                               \
    (da)->items[(da)->count++] = (item);            \
} while(0)
```

This is a **dynamic array** that automatically grows when needed.

### gym_render_nn() - Visualize Network

```c
void gym_render_nn(NN nn, float rx, float ry, float rw, float rh)
{
    // Draws neurons as circles
    // Draws connections as lines
    // Color intensity shows weight strength
    // ...
}
```

### gym_plot() - Draw Cost Graph

```c
void gym_plot(Plot plot, int rx, int ry, int rw, int rh)
{
    // Plots training cost over time
    // Shows learning progress
    // ...
}
```

### gym_process_batch() - Batch Training

```c
void gym_process_batch(Gym_Batch *gb, size_t batch_size,
                       NN nn, NN g, Mat t, float rate)
{
    // Processes training data in mini-batches
    // Updates weights incrementally
    // Tracks cost and progress
    // ...
}
```

---

## Complete Usage Example

```c
#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"
#include "raylib.h"

int main(void)
{
    // Define network architecture: 2 inputs, 3 hidden, 1 output
    size_t arch[] = {2, 3, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));  // Gradient network

    // Initialize with random weights
    nn_rand(nn, -1, 1);

    // Training data for XOR problem
    float train_data[] = {
        0, 0, 0,  // Input: 0,0 → Output: 0
        0, 1, 1,  // Input: 0,1 → Output: 1
        1, 0, 1,  // Input: 1,0 → Output: 1
        1, 1, 0   // Input: 1,1 → Output: 0
    };

    Mat ti = {
        .rows = 4,
        .cols = 2,
        .stride = 3,
        .es = train_data
    };

    Mat to = {
        .rows = 4,
        .cols = 1,
        .stride = 3,
        .es = train_data + 2
    };

    // Training loop
    float learning_rate = 0.1f;
    for(size_t i = 0; i < 10000; ++i) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, learning_rate);

        if(i % 100 == 0) {
            float cost = nn_cost(nn, ti, to);
            printf("Cost: %f\n", cost);
        }
    }

    // Test the trained network
    mat_fill(NN_INPUT(nn), 0);
    MAT_AT(NN_INPUT(nn), 0, 0) = 1;
    MAT_AT(NN_INPUT(nn), 0, 1) = 0;
    nn_forward(nn);

    printf("Prediction for (1,0): %f\n",
           MAT_AT(NN_OUTPUT(nn), 0, 0));

    return 0;
}
```

---

## Learning Checklist

Use this checklist to track your progress through the interactive exercises:

### 🟢 Foundation Skills
- [ ] I understand what a matrix is and how it's stored in memory
- [ ] I can explain what `stride` is and why it's useful
- [ ] I can trace through `MAT_AT(m, i, j)` to find the memory index
- [ ] I implemented the `Mat` structure myself

### 🟡 Matrix Operations
- [ ] I implemented `mat_alloc()` to create a matrix
- [ ] I implemented `mat_fill()` to set all elements
- [ ] I understand matrix multiplication and can compute it by hand
- [ ] I implemented `mat_dot()` for matrix multiplication
- [ ] I implemented `mat_sig()` to apply sigmoid activation

### 🔴 Neural Network
- [ ] I understand the `NN` structure and what each field stores
- [ ] I can draw a diagram of a [2, 3, 1] network
- [ ] I implemented `nn_forward()` to compute predictions
- [ ] I understand what a cost function measures
- [ ] I implemented `nn_cost()` for mean squared error
- [ ] I understand the concept of gradients
- [ ] I implemented `nn_learn()` for gradient descent

### 🏆 Complete Project
- [ ] I wrote a complete program that trains on XOR
- [ ] I can explain how backpropagation works in my own words
- [ ] I experimented with different network architectures
- [ ] I understand the training loop (forward → cost → backward → update)

---

## Common Pitfalls & Debugging Tips

### ⚠️ Common Mistakes Beginners Make

#### 1. Confusing Row-Major vs Column-Major Order
```c
// WRONG (column-major):
index = j * rows + i

// CORRECT (row-major):
index = i * cols + j
```
**Remember:** C uses row-major order!

#### 2. Forgetting to Initialize stride
```c
// WRONG:
Mat m = mat_alloc(2, 3);
m.stride = 2;  // Bug! Should be 3

// CORRECT:
Mat m = mat_alloc(2, 3);
// stride is already set to cols in mat_alloc
```

#### 3. Not Checking Matrix Dimensions Before Operations
```c
// Always assert dimensions match!
void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows);  // Don't forget this!
    NN_ASSERT(dst.cols == a.cols);
    // ... rest of function
}
```

#### 4. Modifying Row Views Affects Parent Matrix
```c
Mat row = mat_row(parent, 0);
MAT_AT(row, 0, 0) = 999;  // This changes parent!
```
**Remember:** Row views share memory with the parent!

#### 5. Learning Rate Too High or Too Low
```c
// Too high - training diverges:
float rate = 10.0f;

// Too low - training is very slow:
float rate = 0.00001f;

// Good starting point:
float rate = 0.1f;
```

#### 6. Forgetting to Seed Random Number Generator
```c
// WRONG - same "random" values every time:
nn_rand(nn, -1, 1);

// CORRECT:
srand(time(NULL));  // Seed once at program start
nn_rand(nn, -1, 1);
```

### 🔍 Debugging Techniques

#### Print Matrix Contents
```c
#define MAT_PRINT(m) mat_print(m, #m, 0)

// Use liberally during development!
MAT_PRINT(my_matrix);
```

#### Check for NaN or Infinity
```c
#include <math.h>

if (isnan(value) || isinf(value)) {
    printf("Found invalid value!\n");
}
```

#### Verify Gradients with Finite Differences
The `nn_finite_diff()` function can verify your backpropagation:
```c
NN nn = nn_alloc(arch, ARRAY_LEN(arch));
NN g = nn_alloc(arch, ARRAY_LEN(arch));
NN g2 = nn_alloc(arch, ARRAY_LEN(arch));

// Compare backprop with numerical gradient
nn_backprop(nn, g, ti, to);
nn_finite_diff(nn, g2, 0.001f, ti, to);

// g and g2 should be similar!
```

#### Track Cost During Training
```c
for (size_t i = 0; i < 10000; ++i) {
    nn_backprop(nn, g, ti, to);
    nn_learn(nn, g, rate);

    if (i % 100 == 0) {
        float cost = nn_cost(nn, ti, to);
        printf("Epoch %zu: Cost = %f\n", i, cost);

        // Cost should decrease over time!
        // If it increases, something is wrong.
    }
}
```

---

## Key Takeaways

1. **Matrix operations** are the foundation of neural network computations
2. **Forward pass** computes predictions: `a = σ(a × W + b)`
3. **Backpropagation** computes gradients using the chain rule
4. **Gradient descent** updates weights: `W = W - α × ∇J`
5. **Cost function** measures prediction error (MSE)
6. **Sigmoid activation** squashes values to (0,1) range
7. **Batch training** processes multiple samples before updating
8. **Visualization** helps understand network behavior

---

## Practice Problems

Test your understanding with these exercises:

### 📝 Beginner Problems

#### Problem 1: Matrix Transpose
Write a function that transposes a matrix (rows become columns):

```c
Mat mat_transpose(Mat m)
{
    // TODO: Return a new matrix that is the transpose of m
    // Hint: For a 2×3 matrix, return a 3×2 matrix
}
```

<details>
<summary>Click for hint</summary>

The result should have dimensions `m.cols × m.rows`.
Element `[i][j]` of the result = element `[j][i]` of the input.
</details>

<details>
<summary>Click for solution</summary>

```c
Mat mat_transpose(Mat m)
{
    Mat result = mat_alloc(m.cols, m.rows);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(result, j, i) = MAT_AT(m, i, j);
        }
    }
    return result;
}
```
</details>

#### Problem 2: Element-wise Multiply
Write a function that multiplies two matrices element-by-element:

```c
void mat_elem_mult(Mat dst, Mat a, Mat b)
{
    // TODO: dst[i][j] = a[i][j] * b[i][j]
}
```

<details>
<summary>Click for solution</summary>

```c
void mat_elem_mult(Mat dst, Mat a, Mat b)
{
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    NN_ASSERT(a.rows == b.rows);
    NN_ASSERT(a.cols == b.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
        }
    }
}
```
</details>

### 📝 Intermediate Problems

#### Problem 3: ReLU Activation
Implement the ReLU activation function and its derivative:

```c
// ReLU: max(0, x)
float reluf(float x)
{
    // TODO: Return x if x > 0, else return 0
}

void mat_relu(Mat m)
{
    // TODO: Apply ReLU to every element
}
```

<details>
<summary>Click for solution</summary>

```c
float reluf(float x)
{
    return x > 0 ? x : 0;
}

void mat_relu(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = reluf(MAT_AT(m, i, j));
        }
    }
}
```
</details>

#### Problem 4: Add a New Layer Function
Write a function that adds a new layer to an existing network:

```c
void nn_add_layer(NN *nn, size_t layer_size)
{
    // TODO: Resize the network and add a new layer
    // This is challenging! Requires reallocation.
}
```

<details>
<summary>Click for hint</summary>

You'll need to:
1. Reallocate the ws, bs, and as arrays
2. Create new matrices for the additional layer
3. Update nn->count
</details>

### 📝 Advanced Problems

#### Problem 5: Implement Momentum
Modify the learning rule to use momentum, which helps escape local minima:

```c
typedef struct {
    NN velocity;  // Stores velocity for each parameter
} MomentumOptimizer;

void nn_learn_momentum(NN nn, NN g, MomentumOptimizer *opt, float rate, float momentum)
{
    // TODO: Update parameters using momentum
    // velocity = momentum * velocity - rate * gradient
    // parameter = parameter + velocity
}
```

#### Problem 6: Implement Dropout
Add dropout regularization to prevent overfitting:

```c
void mat_dropout(Mat m, float rate, bool *mask)
{
    // TODO: Randomly set elements to 0 with probability = rate
    // Store which elements were dropped in mask
}
```

#### Problem 7: Save/Load Neural Network
Write functions to save and load entire neural networks:

```c
void nn_save(FILE *out, NN nn)
{
    // TODO: Save all weights, biases, and architecture
}

NN nn_load(FILE *in)
{
    // TODO: Load and reconstruct a neural network
}
```

---

## Further Learning

- **Matrix multiplication**: Essential for understanding neural networks
- **Calculus**: Derivatives and chain rule for backpropagation
- **Optimization**: Gradient descent variants (SGD, Adam, etc.)
- **Activation functions**: ReLU, tanh, softmax alternatives
- **Regularization**: L1/L2, dropout for preventing overfitting

---

## Summary Table

| Function | Purpose |
|----------|---------|
| `mat_alloc` | Create a new matrix |
| `mat_dot` | Matrix multiplication |
| `mat_sum` | Element-wise addition |
| `mat_sig` | Apply sigmoid activation |
| `mat_fill` | Fill with constant value |
| `mat_copy` | Copy matrix data |
| `mat_rand` | Random initialization |
| `mat_save` | Save to file |
| `mat_load` | Load from file |
| `nn_alloc` | Create neural network |
| `nn_forward` | Compute predictions |
| `nn_backprop` | Calculate gradients |
| `nn_learn` | Update weights |
| `nn_cost` | Compute error |
| `nn_rand` | Randomize weights |

---

## References

- **Neural Networks and Deep Learning** by Michael Nielsen
- **CS231n: Convolutional Neural Networks** (Stanford)
- **The Matrix Cookbook** (matrix reference)

---

*This document provides a comprehensive explanation of the nn.h library, designed for beginners learning neural network implementation in C.*
