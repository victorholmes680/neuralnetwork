# Neural Network Binary Adder - Complete Explanation

## Overview

This program (`adder.c`) implements a **neural network that learns to perform binary addition**. It uses a custom neural network library (`nn.h`) and visualizes the training process using the Raylib graphics library.

### What the Program Does

1. **Trains a neural network** to add two 4-bit binary numbers
2. **Visualizes** the training process in real-time:
   - A cost/loss plot showing how well the network is learning
   - The neural network architecture with weights and activations
   - A verification grid showing all possible inputs and the network's predictions
3. **Allows interactive control** (pause/resume, reset)

---

## Key Concepts

### Binary Addition
- Input: Two 4-bit binary numbers (each can be 0-15, or 0000-1111 in binary)
- Output: Their sum (4-bit result + 1 overflow bit)
- Example: `0101 (5) + 0011 (3) = 1000 (8)`

### Neural Network Training
- **Forward Pass**: Network makes predictions based on inputs
- **Backpropagation**: Calculates how wrong the predictions are
- **Learning**: Adjusts weights to reduce errors over time

---

## Code Breakdown

### 1. Header and Configuration

```c
#define NN_IMPLEMENTATION
#define NN_ENABLE_GYM
#include "nn.h"

#define BITS 4
```

- `NN_IMPLEMENTATION`: Tells `nn.h` to include the actual implementation code
- `NN_ENABLE_GYM`: Enables visualization utilities from the library
- `BITS 4`: We're working with 4-bit binary numbers

### 2. Global Configuration

```c
size_t arch[] = { 2*BITS, 4*BITS, BITS + 1 };  // Network architecture
size_t max_epoch = 100*1000;                    // Maximum training iterations
size_t epochs_per_frame = 103;                  // Training steps per frame
float rate = 1.0f;                              // Learning rate
bool paused = true;                             // Start paused
```

**Network Architecture**: `{8, 16, 5}`
- **Layer 1 (Input)**: 8 neurons (4 bits for first number + 4 bits for second)
- **Layer 2 (Hidden)**: 16 neurons
- **Layer 3 (Output)**: 5 neurons (4 bits for sum + 1 overflow bit)

### 3. The `verify_nn_adder` Function

This function **visualizes all possible additions** the neural network can perform:

```c
void verify_nn_adder(Font font, NN nn, float rx, float ry, float rw, float rh)
```

#### How It Works:

1. **Calculate grid size** based on available screen space:
   ```c
   size_t n = 1 << BITS;  // n = 16 (2^4)
   float cs = s/n;        // Size of each cell
   ```

2. **Loop through all input combinations** (16×16 = 256 combinations):
   ```c
   for(size_t x = 0; x < n; ++x) {      // First number: 0-15
       for(size_t y = 0; y < n; ++y) {  // Second number: 0-15
   ```

3. **Encode inputs as binary** (line 31-34):
   ```c
   for(size_t i = 0; i < BITS; ++i) {
       MAT_AT(NN_INPUT(nn), 0, i) = (x >> i) & 1;        // Extract bit i of x
       MAT_AT(NN_INPUT(nn), 0, i + BITS) = (y >> i) & 1; // Extract bit i of y
   }
   ```
   - `>>` is right-shift (moves bits to lower positions)
   - `& 1` extracts the least significant bit
   - Each number becomes 4 separate inputs (one per bit)

4. **Run forward pass** (line 36):
   ```c
   nn_forward(nn);  // Network makes prediction
   ```

5. **Decode output from binary** (line 38-43):
   ```c
   size_t z = 0.0f;
   for(size_t i = 0; i < BITS; ++i) {
       size_t bit = MAT_AT(NN_OUTPUT(nn), 0, i) > 0.5;
       z = z | (bit << i);  // Set bit i in z
   }
   bool overflow = MAT_AT(NN_OUTPUT(nn), 0, BITS) > 0.5;
   ```
   - Network outputs values between 0 and 1
   - Values > 0.5 are treated as binary "1"
   - `<< i` shifts the bit to the correct position
   - `|` combines all bits into the final number

6. **Draw the result** (line 45-61):
   - Purple cell if overflow detected (sum ≥ 16)
   - White text showing the predicted sum

### 4. The `main` Function

#### Step 1: Generate Training Data (lines 68-110)

```c
size_t n = (1 << BITS);           // 16
size_t rows = n*n;                // 256 (all combinations)
Mat t = mat_alloc(rows, 2 * BITS + BITS + 1);  // 256 × 9 matrix
```

The training matrix has:
- **Columns 0-7**: Inputs (2 numbers × 4 bits each)
- **Columns 8-12**: Expected outputs (4-bit sum + overflow bit)

**Creating the matrix views** (lines 75-88):
```c
Mat ti = { .es = &MAT_AT(t, 0, 0), .rows = t.rows, .cols = 2*BITS, ... };  // Input view
Mat to = { .es = &MAT_AT(t, 0, 2*BITS), .rows = t.rows, .cols = BITS + 1, ... };  // Output view
```
- These are **views** into the same memory, not copies
- `ti` points to the input portion, `to` points to the output portion

**Filling the training data** (lines 92-110):
```c
for(size_t i = 0; i < ti.rows; ++i) {
    size_t x = i / n;        // First number (0-15)
    size_t y = i % n;        // Second number (0-15)
    size_t z = x + y;        // Expected sum

    // Encode inputs
    for(size_t j = 0; j < BITS; ++j) {
        MAT_AT(ti, i, j)         = (x>>j)&1;
        MAT_AT(ti, i, j + BITS)  = (y>>j)&1;
        MAT_AT(to, i, j)         = (z>>j)&1;  // Expected output bits
    }

    // Handle overflow
    if(z >= n) {
        for(size_t j = 0; j < BITS; ++j) {
            MAT_AT(to, i, j) = 1;  // All bits set to 1 for overflow
        }
        MAT_AT(to, i, BITS) = 1;   // Set overflow flag
    } else {
        MAT_AT(to, i, BITS) = 0;   // No overflow
    }
}
```

**Note on overflow handling**: When the sum overflows (≥ 16), the expected output is all 1s (1111) plus the overflow bit set. This is a design choice for handling overflow cases.

#### Step 2: Initialize Neural Network (lines 113-117)

```c
NN nn = nn_alloc(arch, ARRAY_LEN(arch));  // Allocate network
NN g = nn_alloc(arch, ARRAY_LEN(arch));   // Allocate gradient storage
nn_rand(nn, -1, 1);                        // Random weights between -1 and 1
```

- `nn`: The actual neural network
- `g`: Stores gradients (how much to change each weight)

#### Step 3: Initialize Graphics (lines 120-133)

```c
SetConfigFlags(FLAG_WINDOW_RESIZABLE);
InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "xor");
SetTargetFPS(60);
Font font = LoadFontEx("./fonts/iosevka-regular.ttf", 72, NULL, 0);
```

#### Step 4: Main Training Loop (lines 140-181)

```c
while(!WindowShouldClose()) {
    // Handle keyboard input
    if(IsKeyPressed(KEY_SPACE)) paused = !paused;
    if(IsKeyPressed(KEY_R)) {  // Reset
        epoch = 0;
        nn_rand(nn, -1, 1);
        plot.count = 0;
    }

    // Training loop
    for(size_t i = 0; i < epochs_per_frame && !paused && epoch < max_epoch; ++i) {
        nn_backprop(nn, g, ti, to);  // Calculate gradients
        nn_learn(nn, g, rate);       // Update weights
        epoch += 1;
        da_append(&plot, nn_cost(nn, ti, to));  // Record cost
    }

    // Rendering
    BeginDrawing();
    ClearBackground(background_color);

    // Draw three panels: cost plot, network visualization, verification grid
    gym_plot(plot, rx, ry, rw, rh);
    gym_render_nn(nn, rx, ry, rw, rh);
    verify_nn_adder(font, nn, rx, ry, rw, rh);

    // Draw status text
    snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f", ...);
    DrawTextEx(font, buffer, ...);

    EndDrawing();
}
```

---

## Key Functions from `nn.h` Library

| Function | Purpose |
|----------|---------|
| `nn_alloc(arch, count)` | Allocates a neural network with given architecture |
| `nn_rand(nn, min, max)` | Randomizes all weights in the network |
| `nn_forward(nn)` | Computes output from input (forward pass) |
| `nn_backprop(nn, g, ti, to)` | Calculates gradients (backward pass) |
| `nn_learn(nn, g, rate)` | Updates weights based on gradients |
| `nn_cost(nn, ti, to)` | Calculates total error (loss) |
| `MAT_AT(mat, i, j)` | Accesses element at row i, column j |
| `NN_INPUT(nn)` | Macro for input layer matrix |
| `NN_OUTPUT(nn)` | Macro for output layer matrix |

---

## Understanding Matrix Operations

### Matrix Macro: `MAT_AT`

```c
MAT_AT(matrix, row, col) = value;
```

This is a macro that calculates the memory offset for a 2D matrix stored in 1D memory:
```c
#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
```

### Matrix View (Pointer Sharing)

```c
Mat big_matrix = mat_alloc(10, 10);     // 10×10 matrix
Mat view = {
    .es = &MAT_AT(big_matrix, 2, 2),    // Points to element (2,2)
    .rows = 5,
    .cols = 5,
    .stride = big_matrix.stride
};
```
- `view` shares memory with `big_matrix`
- Changes to `view` affect `big_matrix`
- Used to split one matrix into input and output portions

---

## Binary Operations Explained

### Extracting Bits

```c
size_t x = 13;  // Binary: 1101
(x >> 0) & 1;   // = 1 (least significant bit)
(x >> 1) & 1;   // = 0
(x >> 2) & 1;   // = 1
(x >> 3) & 1;   // = 1 (most significant bit)
```

### Setting Bits

```c
size_t z = 0;
z |= (1 << 0);  // Set bit 0: z = 0001
z |= (1 << 2);  // Set bit 2: z = 0101
```

---

## How the Training Works

### The Training Process

1. **Forward Pass**: Network makes prediction → likely wrong initially
2. **Calculate Cost**: Compare prediction to actual answer
3. **Backpropagation**: Calculate gradients (which weights contributed to error)
4. **Update Weights**: Adjust weights slightly to reduce error
5. **Repeat**: Do this thousands of times

### Learning Rate

```c
float rate = 1.0f;
```

- **Too high**: Training might become unstable
- **Too low**: Training takes too long
- **Just right**: Steady improvement

### Cost Function

The cost (loss) measures how wrong the network is:
- **High cost**: Network predictions are very wrong
- **Low cost**: Network predictions are close to correct
- **Goal**: Minimize cost over training

---

## Interactive Controls

| Key | Action |
|-----|--------|
| SPACE | Pause/Resume training |
| R | Reset network (randomize weights and restart) |

---

## Visual Output Layout

The window is divided into **three equal panels**:

```
┌─────────────┬─────────────┬─────────────┐
│             │             │             │
│   Cost      │   Neural    │  Verify     │
│   Plot      │   Network   │  Grid       │
│             │   Diagram   │  (16×16)    │
│             │             │             │
└─────────────┴─────────────┴─────────────┘
```

1. **Left Panel**: Cost over time (should decrease)
2. **Middle Panel**: Neural network visualization
   - Shows weights (line thickness/color)
   - Shows activations (colored circles)
3. **Right Panel**: Verification grid
   - Each cell shows one addition
   - Position = inputs (x + y)
   - Text = network's prediction
   - Purple = overflow detected

---

## Important Notes for Beginners

### Memory Management
- The program allocates memory for matrices and networks
- In this simple program, cleanup is handled by OS on exit
- In production code, you'd need explicit cleanup functions

### Float vs Size_t
- `float`: For neural network weights and activations
- `size_t`: For indices, counts, and binary values

### Training Time
- Early: High cost, random predictions
- Mid: Cost decreases, some predictions correct
- Late: Low cost, most/all predictions correct

### Why This Works
- Neural networks are **universal function approximators**
- Given enough neurons and training, they can learn any computable function
- Binary addition is a simple, deterministic function → easily learned

---

## Summary

This program demonstrates:
1. **Neural network training** from scratch
2. **Binary data encoding** for neural networks
3. **Real-time visualization** of training progress
4. **Interactive control** of the training process

The network learns to perform 4-bit binary addition entirely from examples—no explicit addition rules are programmed, only the training data (input-output pairs) and the learning algorithm.

---

## Further Reading

- **Backpropagation**: How gradients are calculated
- **Activation Functions**: What happens inside each neuron
- **Matrix Operations**: The math behind neural networks
- **Hyperparameters**: How architecture affects learning
