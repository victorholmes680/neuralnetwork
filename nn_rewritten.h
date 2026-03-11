// =============================================================================
// nn_rewritten.h - A Neural Network Library in C
// =============================================================================
// This is a complete rewrite of nn.h with extensive comments for learning
// purposes. It implements a simple feedforward neural network with
// backpropagation training.
//
// USAGE:
//   1. Define NN_IMPLEMENTATION in exactly ONE source file before including
//   2. Optionally define NN_ENABLE_GYM for visualization features
//   3. Include this header in other files WITHOUT NN_IMPLEMENTATION
//
// EXAMPLE:
//   // In main.c:
//   #define NN_IMPLEMENTATION
//   #include "nn_rewritten.h"
//
//   // In other files:
//   #include "nn_rewritten.h"
// =============================================================================

#ifndef NN_REWRITTEN_H_
#define NN_REWRITTEN_H_

// =============================================================================
// SECTION 1: STANDARD LIBRARY INCLUDES
// =============================================================================

#include <stdbool.h>    // Provides: bool, true, false
#include <stddef.h>     // Provides: size_t (unsigned integer type for sizes)
#include <stdio.h>      // Provides: FILE, printf, fread, fwrite
#include <time.h>       // Provides: time (for seeding random number generator)
#include <math.h>       // Provides: expf (for sigmoid function)
#include <stdint.h>     // Provides: uint64_t (for file I/O)
#include <float.h>      // Provides: FLT_MAX, FLT_MIN (for plotting)
#include <string.h>     // Provides: strlen, memcpy

// =============================================================================
// SECTION 2: CUSTOMIZABLE MACROS
// =============================================================================

// NN_MALLOC allows users to customize memory allocation
// This is useful for:
//   - Debugging (tracking allocations)
//   - Memory pools (custom allocators)
//   - Embedded systems (special memory management)
#ifndef NN_MALLOC
    #include <stdlib.h>   // Standard library for malloc
    #define NN_MALLOC malloc
#endif

// NN_ASSERT allows users to customize assertion behavior
// Useful for:
//   - Custom error handling
//   - Logging assertions
//   - Release builds with different behavior
#ifndef NN_ASSERT
    #include <assert.h>   // Standard library for assert
    #define NN_ASSERT assert
#endif

// ARRAY_LEN calculates the number of elements in a static array
// Works by dividing total size by size of first element
// Example: ARRAY_LEN(arr) where arr is {1,2,3} returns 3
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

// =============================================================================
// SECTION 3: DATA STRUCTURES
// =============================================================================

// -----------------------------------------------------------------------------
// Gym_Batch: Tracks batch processing during training
// -----------------------------------------------------------------------------
// When training neural networks, we often process data in "batches" rather
// than all at once. This structure tracks the progress of batch processing.
//
// Fields:
//   - begin: Index of the first sample in the current batch
//   - cost:  Accumulated error (cost) for all batches processed so far
//   - finished: Boolean flag indicating if all batches are complete
typedef struct {
    size_t begin;      // Starting index for current batch
    float cost;        // Running total of cost across batches
    bool finished;     // true when all batches processed
} Gym_Batch;

// -----------------------------------------------------------------------------
// Mat: Matrix structure for 2D arrays of floats
// -----------------------------------------------------------------------------
// A matrix is a 2D grid of numbers. This is the fundamental data structure
// for neural network computations.
//
// Fields:
//   - rows:   Number of rows in the matrix
//   - cols:   Number of columns in the matrix
//   - stride: Number of elements to skip to get to the next row
//             (usually equals cols, but allows for efficient row views)
//   - es:     Pointer to the actual data stored in row-major order
//
// Memory Layout Example (for a 2x3 matrix):
//   es = [1, 2, 3, 4, 5, 6]
//   This represents:
//     [1, 2, 3]
//     [4, 5, 6]
//
// The stride allows us to create "views" of rows that share memory with
// the parent matrix, avoiding expensive memory copies.
typedef struct {
    size_t rows;    // Number of rows
    size_t cols;    // Number of columns
    size_t stride;  // Elements per row (enables row views)
    float *es;      // Pointer to matrix elements (row-major order)
} Mat;

// -----------------------------------------------------------------------------
// MAT_AT: Access matrix element at row i, column j
// -----------------------------------------------------------------------------
// This macro calculates the memory index for element (i, j).
//
// Formula: index = i * stride + j
//
// Why stride instead of cols?
//   - For normal matrices: stride == cols
//   - For row views: stride == parent.stride (can be larger!)
//   - This enables zero-copy row operations
//
// Example:
//   Mat m = {...};
//   float value = MAT_AT(m, 1, 2);  // Get element at row 1, column 2
//   MAT_AT(m, 0, 0) = 5.0f;         // Set element at row 0, column 0
#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]

// -----------------------------------------------------------------------------
// Matrix Function Declarations
// -----------------------------------------------------------------------------

// mat_alloc: Allocate a new matrix with given dimensions
// Parameters:
//   - rows: Number of rows
//   - cols: Number of columns
// Returns: A new Mat structure with allocated memory
Mat mat_alloc(size_t rows, size_t cols);

// mat_save: Save matrix to a file
// Parameters:
//   - out: Open file pointer for writing
//   - m:   Matrix to save
void mat_save(FILE *out, Mat m);

// mat_load: Load matrix from a file
// Parameters:
//   - in: Open file pointer for reading
// Returns: The loaded matrix
Mat mat_load(FILE *in);

// mat_fill: Fill all elements of a matrix with a value
// Parameters:
//   - m: Matrix to fill
//   - x: Value to fill with
void mat_fill(Mat m, float x);

// mat_rand: Fill matrix with random values in range [low, high]
// Parameters:
//   - m:    Matrix to fill
//   - low:  Minimum random value
//   - high: Maximum random value
void mat_rand(Mat m, float low, float high);

// mat_row: Create a 1xN matrix view of a single row
// Parameters:
//   - m:   Parent matrix
//   - row: Row index to view
// Returns: A 1-row matrix that shares memory with parent
Mat mat_row(Mat m, size_t row);

// mat_copy: Copy data from source matrix to destination
// Parameters:
//   - dst: Destination matrix (must match src dimensions)
//   - src: Source matrix
void mat_copy(Mat dst, Mat src);

// mat_dot: Matrix multiplication (dot product)
// Parameters:
//   - dst: Destination matrix (must be rows=a.rows, cols=b.cols)
//   - a:   First matrix (m×n)
//   - b:   Second matrix (n×p)
// Result: dst = a × b (produces m×p matrix)
void mat_dot(Mat dst, Mat a, Mat b);

// mat_sum: Element-wise addition (adds a to dst)
// Parameters:
//   - dst: Destination matrix (dst += a)
//   - a:   Matrix to add (must match dst dimensions)
void mat_sum(Mat dst, Mat a);

// mat_sig: Apply sigmoid activation to all elements
// Parameters:
//   - m: Matrix to transform (each element becomes sigmoid(element))
// The sigmoid function: σ(x) = 1 / (1 + e^(-x))
void mat_sig(Mat m);

// mat_print: Print matrix to console for debugging
// Parameters:
//   - m:       Matrix to print
//   - name:    Label to show before matrix
//   - padding: Number of spaces to indent
void mat_print(Mat m, const char *name, size_t padding);

// mat_shuffle_rows: Randomly shuffle rows of matrix
// Parameters:
//   - m: Matrix to shuffle (rows are permuted, columns preserved)
void mat_shuffle_rows(Mat m);

// MAT_PRINT: Convenience macro to print matrix with its variable name
// Example: MAT_PRINT(my_matrix) prints "my_matrix = [...]"
#define MAT_PRINT(m) mat_print(m, #m, 0)

// -----------------------------------------------------------------------------
// NN: Neural Network Structure
// -----------------------------------------------------------------------------
// A feedforward neural network with multiple layers.
//
// Structure:
//   - Input layer:  Receives initial data
//   - Hidden layers: Process data (can have multiple)
//   - Output layer: Produces final prediction
//
// Each connection between layers has:
//   - Weights (ws): How strongly each neuron connects
//   - Biases (bs): Offset values for each neuron
//   - Activations (as): Output values after activation function
//
// Fields:
//   - count: Number of layers with learnable parameters (excluding input)
//   - ws:    Array of weight matrices (ws[i] connects layer i to i+1)
//   - bs:    Array of bias matrices (bs[i] is bias for layer i+1)
//   - as:    Array of activation matrices (as[i] is activation of layer i)
//
// Architecture Example [2, 3, 1]:
//   - Input layer:  2 neurons
//   - Hidden layer: 3 neurons
//   - Output layer: 1 neuron
//
//   - count = 2 (two layers with parameters)
//   - ws[0] = 2×3 matrix (input→hidden weights)
//   - bs[0] = 1×3 matrix (hidden biases)
//   - ws[1] = 3×1 matrix (hidden→output weights)
//   - bs[1] = 1×1 matrix (output bias)
//   - as[0] = 1×2 matrix (input activations)
//   - as[1] = 1×3 matrix (hidden activations)
//   - as[2] = 1×1 matrix (output activations)
typedef struct {
    size_t count;  // Number of weight matrices (layers with parameters)
    Mat *ws;       // Array of weight matrices
    Mat *bs;       // Array of bias matrices
    Mat *as;       // Array of activation matrices (includes input)
} NN;

// -----------------------------------------------------------------------------
// Neural Network Helper Macros
// -----------------------------------------------------------------------------

// NN_INPUT: Get the input layer activation matrix
// This is as[0], where we set the input data before forward pass
#define NN_INPUT(nn) (nn).as[0]

// NN_OUTPUT: Get the output layer activation matrix
// This is as[count], the final output after forward pass
#define NN_OUTPUT(nn) (nn).as[(nn).count]

// -----------------------------------------------------------------------------
// Neural Network Function Declarations
// -----------------------------------------------------------------------------

// nn_alloc: Create a neural network with given architecture
// Parameters:
//   - arch:        Array of layer sizes (e.g., {2, 3, 1} for 2-3-1 network)
//   - arch_count:  Number of elements in arch array
// Returns: Allocated and initialized neural network
NN nn_alloc(size_t *arch, size_t arch_count);

// nn_zero: Reset all weights, biases, and activations to zero
// Parameters:
//   - nn: Neural network to zero out
void nn_zero(NN nn);

// nn_backprop: Calculate gradients using backpropagation
// Parameters:
//   - nn:  Neural network to compute gradients for
//   - g:   Gradient network (stores computed gradients)
//   - ti:  Training inputs matrix
//   - to:  Training outputs matrix (target values)
//
// This computes how much each parameter contributes to the error,
// storing the results in the gradient network g.
void nn_backprop(NN nn, NN g, Mat ti, Mat to);

// nn_print: Print neural network structure for debugging
// Parameters:
//   - nn:   Neural network to print
//   - name: Label to show before network
void nn_print(NN nn, const char *name);

// NN_PRINT: Convenience macro to print network with its variable name
#define NN_PRINT(nn) nn_print(nn, #nn)

// nn_rand: Initialize all weights and biases with random values
// Parameters:
//   - nn:   Neural network to randomize
//   - low:  Minimum random value
//   - high: Maximum random value
void nn_rand(NN nn, float low, float high);

// nn_forward: Perform forward pass (compute predictions)
// Parameters:
//   - nn: Neural network to run
//
// This takes the input from NN_INPUT(nn), propagates it through
// all layers, and stores the result in NN_OUTPUT(nn).
void nn_forward(NN nn);

// nn_cost: Calculate the mean squared error of the network
// Parameters:
//   - nn:  Neural network to evaluate
//   - ti:  Training inputs
//   - to:  Target outputs
// Returns: The average squared error across all samples
//
// Lower cost means better predictions. This is the function
// we're trying to minimize during training.
float nn_cost(NN nn, Mat ti, Mat to);

// nn_finite_diff: Compute gradients numerically (for testing)
// Parameters:
//   - nn:   Neural network
//   - g:    Gradient network (stores computed gradients)
//   - eps:  Small value for numerical differentiation
//   - ti:   Training inputs
//   - to:   Target outputs
//
// This is an alternative to backpropagation that computes gradients
// by slightly perturbing each parameter. It's slower but useful
// for verifying that backpropagation is implemented correctly.
void nn_finite_diff(NN nn, NN g, float eps, Mat ti, Mat to);

// nn_learn: Update network parameters using computed gradients
// Parameters:
//   - nn:   Neural network to update
//   - g:    Gradient network (contains gradients)
//   - rate: Learning rate (step size for updates)
//
// This implements gradient descent: parameter = parameter - rate * gradient
void nn_learn(NN nn, NN g, float rate);

// =============================================================================
// SECTION 4: OPTIONAL VISUALIZATION SYSTEM (GYM)
// =============================================================================

// The gym system provides visualization using the Raylib library.
// It's only included if NN_ENABLE_GYM is defined before including this file.
#ifdef NN_ENABLE_GYM

    #include "raylib.h"  // Raylib library for visualization

    // -----------------------------------------------------------------------------
    // Plot: Dynamic array for storing values to graph
    // -----------------------------------------------------------------------------
    // This structure stores a sequence of values (like cost over time)
    // that can be plotted as a graph.
    //
    // Fields:
    //   - items:    Pointer to array of float values
    //   - count:    Current number of values in array
    //   - capacity: Total allocated capacity (can grow dynamically)
    typedef struct {
        float *items;      // Array of values
        size_t count;      // Number of values stored
        size_t capacity;   // Allocated capacity
    } Plot;

    // Initial capacity when creating a new plot
    #define DA_INIT_CAP 256

    // -----------------------------------------------------------------------------
    // da_append: Dynamic array append macro
    // -----------------------------------------------------------------------------
    // This macro adds an item to a dynamic array, automatically growing
    // the array if necessary.
    //
    // Parameters:
    //   - da:   Pointer to Plot structure
    //   - item: Value to append
    //
    // The macro uses a do-while(0) pattern to make it safe to use in
    // if statements without curly braces.
    //
    // Growth strategy: Double the capacity when full (amortized O(1))
    #define da_append(da, item)                                \
    do {                                                        \
        if (((da))->count >= ((da))->capacity) {                \
            (da)->capacity = ((da)->capacity == 0) ?            \
                DA_INIT_CAP : (da)->capacity * 2;               \
            (da)->items = realloc((da)->items,                  \
                (da)->capacity * sizeof(*(da)->items));         \
            NN_ASSERT((da)->items != NULL &&                    \
                "Buy more RAM lol");                            \
        }                                                       \
        (da)->items[(da)->count++] = (item);                    \
    } while(0)

    // -----------------------------------------------------------------------------
    // Gym Function Declarations
    // -----------------------------------------------------------------------------

    // gym_render_nn: Visualize the neural network structure
    // Parameters:
    //   - nn:  Neural network to visualize
    //   - rx:  X position of rendering area
    //   - ry:  Y position of rendering area
    //   - rw:  Width of rendering area
    //   - rh:  Height of rendering area
    //
    // Draws:
    //   - Neurons as circles (colored by bias strength)
    //   - Connections as lines (colored by weight strength)
    //   - Input layer on left, output layer on right
    void gym_render_nn(NN nn, float rx, float ry, float rw, float rh);

    // gym_plot: Plot a sequence of values as a line graph
    // Parameters:
    //   - plot: Plot structure containing values to graph
    //   - rx:   X position of plot area
    //   - ry:   Y position of plot area
    //   - rw:   Width of plot area
    //   - rh:   Height of plot area
    //
    // Draws a line graph showing values over time (e.g., training cost)
    void gym_plot(Plot plot, int rx, int ry, int rw, int rh);

    // gym_process_batch: Process one batch of training data
    // Parameters:
    //   - gb:         Batch tracker (stores progress)
    //   - batch_size: Number of samples per batch
    //   - nn:         Neural network to train
    //   - g:          Gradient network
    //   - t:          Training data matrix (inputs and outputs)
    //   - rate:       Learning rate
    //
    // This processes one mini-batch of training data, updating the
    // network weights and tracking the cost. It's useful for
    // showing training progress incrementally.
    void gym_process_batch(Gym_Batch *gb, size_t batch_size,
                           NN nn, NN g, Mat t, float rate);

#endif // NN_ENABLE_GYM

#endif // NN_REWRITTEN_H_

// =============================================================================
// SECTION 5: IMPLEMENTATION
// =============================================================================
// The implementation is only compiled if NN_IMPLEMENTATION is defined.
// This pattern allows the header to be included in multiple files while
// the implementation is only compiled once.
// =============================================================================

#ifdef NN_IMPLEMENTATION

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// -----------------------------------------------------------------------------
// sigmoidf: Sigmoid activation function
// -----------------------------------------------------------------------------
// The sigmoid function squashes any input to the range (0, 1).
// It's commonly used in neural networks because:
//   1. Output is bounded (useful for probabilities)
//   2. It's differentiable (needed for backpropagation)
//   3. The derivative is simple: σ'(x) = σ(x) * (1 - σ(x))
//
// Formula: σ(x) = 1 / (1 + e^(-x))
//
// Properties:
//   - As x → +∞, σ(x) → 1
//   - As x → -∞, σ(x) → 0
//   - σ(0) = 0.5
//   - Symmetric around (0, 0.5)
//
static float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// -----------------------------------------------------------------------------
// rand_float: Generate random float in range [0, 1]
// -----------------------------------------------------------------------------
// Uses the standard library's rand() function which returns an integer
// between 0 and RAND_MAX. We divide by RAND_MAX to get a float in [0, 1].
//
static float rand_float(void)
{
    return (float)rand() / (float)RAND_MAX;
}

// =============================================================================
// MATRIX IMPLEMENTATIONS
// =============================================================================

// -----------------------------------------------------------------------------
// mat_alloc: Create and allocate a new matrix
// -----------------------------------------------------------------------------
Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;  // For new matrices, stride equals cols
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL && "Memory allocation failed");
    return m;
}

// -----------------------------------------------------------------------------
// mat_print: Print matrix contents for debugging
// -----------------------------------------------------------------------------
void mat_print(Mat m, const char *name, size_t padding)
{
    printf("%*s%s = [\n", (int)padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", (int)padding, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", (int)padding, "");
}

// -----------------------------------------------------------------------------
// mat_fill: Fill all elements with a constant value
// -----------------------------------------------------------------------------
void mat_fill(Mat m, float x)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = x;
        }
    }
}

// -----------------------------------------------------------------------------
// mat_dot: Matrix multiplication (C = A × B)
// -----------------------------------------------------------------------------
// Matrix multiplication is the core operation in neural networks.
// It computes the weighted sum of inputs.
//
// For matrices A (m×n) and B (n×p), the product C (m×p) is:
//   C[i][j] = Σ(A[i][k] * B[k][j]) for k = 0 to n-1
//
// Geometric interpretation:
//   - Each element C[i][j] is the dot product of row i of A
//     and column j of B
//   - This computes how much row i "matches" column j
//
void mat_dot(Mat dst, Mat a, Mat b)
{
    // Validate dimensions
    NN_ASSERT(a.cols == b.rows && "Matrix dimensions incompatible");
    NN_ASSERT(dst.rows == a.rows && "Destination row count mismatch");
    NN_ASSERT(dst.cols == b.cols && "Destination column count mismatch");

    size_t n = a.cols;  // Number of multiplications per element

    // Compute each element of the result matrix
    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = 0;  // Initialize to zero
            for (size_t k = 0; k < n; ++k) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// mat_row: Create a view of a single row
// -----------------------------------------------------------------------------
// This creates a 1×cols matrix that points to a row in the parent matrix.
// It SHARES memory with the parent, so changes affect both!
//
// Why share memory?
//   - Avoids expensive memory copies
//   - Allows efficient row-wise operations
//   - Essential for batch processing
//
Mat mat_row(Mat m, size_t row)
{
    return (Mat){
        .rows = 1,
        .cols = m.cols,
        .stride = m.stride,  // Share stride with parent
        .es = &MAT_AT(m, row, 0),  // Point to row's first element
    };
}

// -----------------------------------------------------------------------------
// mat_copy: Copy matrix data from src to dst
// -----------------------------------------------------------------------------
void mat_copy(Mat dst, Mat src)
{
    NN_ASSERT(dst.rows == src.rows && "Row count mismatch");
    NN_ASSERT(dst.cols == src.cols && "Column count mismatch");

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// mat_sum: Element-wise addition (dst += a)
// -----------------------------------------------------------------------------
void mat_sum(Mat dst, Mat a)
{
    NN_ASSERT(dst.rows == a.rows && "Row count mismatch");
    NN_ASSERT(dst.cols == a.cols && "Column count mismatch");

    for (size_t i = 0; i < dst.rows; ++i) {
        for (size_t j = 0; j < dst.cols; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

// -----------------------------------------------------------------------------
// mat_sig: Apply sigmoid activation to all elements
// -----------------------------------------------------------------------------
// Transforms each element x to σ(x) = 1/(1 + e^(-x))
// This squashes values to the range (0, 1).
//
void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

// -----------------------------------------------------------------------------
// mat_rand: Fill matrix with random values
// -----------------------------------------------------------------------------
void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            // Generate random float in [low, high]
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

// -----------------------------------------------------------------------------
// mat_save: Save matrix to file
// -----------------------------------------------------------------------------
// File format:
//   - Magic number: "nn.h.mat" (8 bytes) - identifies file type
//   - Rows: size_t (variable size, typically 8 bytes)
//   - Cols: size_t (variable size, typically 8 bytes)
//   - Data: rows * cols floats (4 bytes each)
//
void mat_save(FILE *out, Mat m)
{
    const char *magic = "nn.h.mat";
    fwrite(magic, strlen(magic), 1, out);
    fwrite(&m.rows, sizeof(m.rows), 1, out);
    fwrite(&m.cols, sizeof(m.cols), 1, out);

    // Write one row at a time, handling partial writes
    for (size_t i = 0; i < m.rows; ++i) {
        size_t n = fwrite(&MAT_AT(m, i, 0), sizeof(*m.es), m.cols, out);
        // Continue writing if not all bytes were written
        while (n < m.cols && !ferror(out)) {
            size_t k = fwrite(m.es + n, sizeof(*m.es), m.cols - n, out);
            n += k;
        }
    }
}

// -----------------------------------------------------------------------------
// mat_load: Load matrix from file
// -----------------------------------------------------------------------------
Mat mat_load(FILE *in)
{
    uint64_t magic;
    size_t _;  // Dummy variable for fread return value
    _ = fread(&magic, sizeof(magic), 1, in);
    // TODO: Verify magic number

    size_t rows, cols;
    _ = fread(&rows, sizeof(rows), 1, in);
    _ = fread(&cols, sizeof(cols), 1, in);
    (void)_;  // Suppress unused variable warning

    Mat m = mat_alloc(rows, cols);

    // Read all data, handling partial reads
    size_t n = fread(m.es, sizeof(*m.es), rows * cols, in);
    while (n < rows * cols && !ferror(in)) {
        size_t k = fread(m.es + n, sizeof(*m.es), rows * cols - n, in);
        n += k;
    }
    return m;
}

// -----------------------------------------------------------------------------
// mat_shuffle_rows: Randomly shuffle matrix rows
// -----------------------------------------------------------------------------
// Uses the Fisher-Yates shuffle algorithm for O(n) time complexity.
// This is important for training to prevent the network from learning
// the order of the training data.
//
void mat_shuffle_rows(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        // Pick random row from i to end
        size_t j = i + rand() % (m.rows - i);
        if (i != j) {
            // Swap rows i and j
            for (size_t k = 0; k < m.cols; ++k) {
                float t = MAT_AT(m, i, k);
                MAT_AT(m, i, k) = MAT_AT(m, j, k);
                MAT_AT(m, j, k) = t;
            }
        }
    }
}

// =============================================================================
// NEURAL NETWORK IMPLEMENTATIONS
// =============================================================================

// -----------------------------------------------------------------------------
// nn_alloc: Create a neural network with given architecture
// -----------------------------------------------------------------------------
// The architecture is specified as an array of layer sizes.
// For example, arch = {2, 3, 1} creates:
//   - Input layer: 2 neurons
//   - Hidden layer: 3 neurons
//   - Output layer: 1 neuron
//
NN nn_alloc(size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0 && "Architecture must have at least 1 layer");

    NN nn;
    nn.count = arch_count - 1;  // Number of layers with parameters

    // Allocate arrays for weights, biases, and activations
    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(nn.ws != NULL && "Weight allocation failed");

    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(nn.bs != NULL && "Bias allocation failed");

    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL && "Activation allocation failed");

    // Allocate input layer (no weights for input layer)
    nn.as[0] = mat_alloc(1, arch[0]);

    // Allocate remaining layers
    for (size_t i = 1; i < arch_count; ++i) {
        // Weights: from previous layer cols to current layer size
        nn.ws[i - 1] = mat_alloc(nn.as[i - 1].cols, arch[i]);
        // Biases: 1 row, current layer size columns
        nn.bs[i - 1] = mat_alloc(nn.as[i - 1].rows, arch[i]);
        // Activations: same as biases dimensions
        nn.as[i] = mat_alloc(nn.as[i - 1].rows, arch[i]);
    }

    return nn;
}

// -----------------------------------------------------------------------------
// nn_zero: Reset all network parameters to zero
// -----------------------------------------------------------------------------
void nn_zero(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
        mat_fill(nn.as[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}

// -----------------------------------------------------------------------------
// nn_forward: Compute predictions (forward pass)
// -----------------------------------------------------------------------------
// The forward pass propagates input through the network to produce output.
//
// For each layer l:
//   1. Weighted sum: y = as[l] × ws[l]
//   2. Add bias: y = y + bs[l]
//   3. Activation: as[l+1] = σ(y)
//
// Mathematical notation:
//   a^(l+1) = σ(a^(l) × W^(l) + b^(l))
//
void nn_forward(NN nn)
{
    for (size_t i = 0; i < nn.count; ++i) {
        // Step 1: Multiply by weights: a × W
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        // Step 2: Add bias: y + b
        mat_sum(nn.as[i + 1], nn.bs[i]);
        // Step 3: Apply activation: σ(y)
        mat_sig(nn.as[i + 1]);
    }
}

// -----------------------------------------------------------------------------
// nn_cost: Calculate mean squared error
// -----------------------------------------------------------------------------
// The cost function measures how wrong the network's predictions are.
// We use Mean Squared Error (MSE):
//
//   J = (1/n) × Σ(predicted - actual)²
//
// Where:
//   - n: number of training samples
//   - predicted: network output
//   - actual: target value
//
// Lower cost = better predictions. Training aims to minimize this.
//
float nn_cost(NN nn, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows && "Input and output row count mismatch");
    NN_ASSERT(to.cols == NN_OUTPUT(nn).cols && "Output dimension mismatch");

    size_t n = ti.rows;  // Number of training samples
    float c = 0;

    // Compute error for each sample
    for (size_t i = 0; i < n; ++i) {
        Mat x = mat_row(ti, i);  // Get input sample
        Mat y = mat_row(to, i);  // Get target output

        mat_copy(NN_INPUT(nn), x);  // Set input
        nn_forward(nn);             // Compute prediction

        // Sum squared errors across all outputs
        size_t q = to.cols;
        for (size_t j = 0; j < q; ++j) {
            float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
            c += d * d;  // Squared error
        }
    }
    return c / n;  // Mean squared error
}

// -----------------------------------------------------------------------------
// nn_backprop: Calculate gradients using backpropagation
// -----------------------------------------------------------------------------
// Backpropagation computes the gradient of the cost function with respect
// to each parameter. It uses the chain rule to propagate errors backward
// through the network.
//
// Chain rule for neural networks:
//   ∂J/∂W = ∂J/∂a × ∂a/∂y × ∂y/∂W
//
// For sigmoid activation: σ'(x) = σ(x) × (1 - σ(x))
//
// This implementation accumulates gradients across all training samples
// and averages them at the end.
//
void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    NN_ASSERT(ti.rows == to.rows && "Input and output row count mismatch");
    size_t n = ti.rows;  // Number of training samples
    NN_ASSERT(NN_OUTPUT(nn).cols == to.cols && "Output dimension mismatch");

    nn_zero(g);  // Reset gradients to zero

    // Process each training sample
    for (size_t i = 0; i < n; ++i) {
        // Set input and compute forward pass
        mat_copy(NN_INPUT(nn), mat_row(ti, i));
        nn_forward(nn);

        // Reset activation gradients for this sample
        for (size_t j = 0; j <= nn.count; ++j) {
            mat_fill(g.as[j], 0);
        }

        // Calculate output layer error (difference from target)
        for (size_t j = 0; j < to.cols; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) =
                MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        // Backpropagate error through layers
        for (size_t l = nn.count; l > 0; --l) {
            for (size_t j = 0; j < nn.as[l].cols; ++j) {
                float a = MAT_AT(nn.as[l], 0, j);      // Activation value
                float da = MAT_AT(g.as[l], 0, j);       // Error gradient

                // Bias gradient: ∂J/∂b = 2 × da × a × (1 - a)
                MAT_AT(g.bs[l - 1], 0, j) += 2 * da * a * (1 - a);

                for (size_t k = 0; k < nn.as[l - 1].cols; ++k) {
                    float pa = MAT_AT(nn.as[l - 1], 0, k);  // Previous activation
                    float w = MAT_AT(nn.ws[l - 1], k, j);    // Weight value

                    // Weight gradient: ∂J/∂W = 2 × da × a × (1 - a) × pa
                    MAT_AT(g.ws[l - 1], k, j) += 2 * da * a * (1 - a) * pa;

                    // Propagate error to previous layer: ∂J/∂pa
                    MAT_AT(g.as[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
                }
            }
        }
    }

    // Average gradients across all samples
    for (size_t i = 0; i < g.count; ++i) {
        for (size_t j = 0; j < g.ws[i].rows; ++j) {
            for (size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].rows; ++j) {
            for (size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// nn_learn: Update network parameters using gradient descent
// -----------------------------------------------------------------------------
// Gradient descent updates parameters in the direction that reduces cost:
//
//   parameter = parameter - learning_rate × gradient
//
// The learning rate controls how big of a step we take:
//   - Too small: slow learning
//   - Too large: may overshoot or diverge
//
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

// -----------------------------------------------------------------------------
// nn_rand: Initialize network with random weights and biases
// -----------------------------------------------------------------------------
// Random initialization is crucial for breaking symmetry and allowing
// the network to learn different features in different neurons.
//
void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
    }
}

// -----------------------------------------------------------------------------
// nn_print: Print network structure for debugging
// -----------------------------------------------------------------------------
void nn_print(NN nn, const char *name)
{
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
        snprintf(buf, sizeof(buf), "ws%zu", i);
        mat_print(nn.ws[i], buf, 4);
        snprintf(buf, sizeof(buf), "bs%zu", i);
        mat_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

// =============================================================================
// GYM VISUALIZATION IMPLEMENTATIONS
// =============================================================================
#ifdef NN_ENABLE_GYM

// -----------------------------------------------------------------------------
// gym_render_nn: Visualize neural network structure
// -----------------------------------------------------------------------------
// This draws a visual representation of the neural network with:
//   - Neurons as circles (size based on rh parameter)
//   - Connections as lines between neurons
//   - Colors representing weight/bias strength
//
// Color scheme:
//   - Low values: magenta/purple
//   - High values: green
//   - Intensity = sigmoid(value)
//
void gym_render_nn(NN nn, float rx, float ry, float rw, float rh)
{
    Color low_color = {0xFF, 0x00, 0xFF, 0xFF};  // Magenta
    Color high_color = {0x00, 0xFF, 0x00, 0xFF}; // Green

    float neuron_radius = rh * 0.03f;           // Size of neuron circles
    float layer_border_vpad = rh * 0.08f;       // Vertical padding
    float layer_border_hpad = rh * 0.06f;       // Horizontal padding

    // Calculate dimensions for the network visualization
    float nn_width = rw - 2 * layer_border_hpad;
    float nn_height = rh - 2 * layer_border_vpad;

    // Center the network in the rendering area
    float nn_x = rx + rw / 2 - nn_width / 2;
    float nn_y = ry + rh / 2 - nn_height / 2;

    size_t arch_count = nn.count + 1;  // Total layers including input
    float layer_hpad = nn_width / arch_count;  // Width per layer

    // Draw each layer
    for (size_t l = 0; l < arch_count; ++l) {
        float layer_vpad1 = nn_height / nn.as[l].cols;  // Height per neuron

        for (size_t i = 0; i < nn.as[l].cols; ++i) {
            float cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
            float cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;

            // Draw connections to next layer
            if (l + 1 < arch_count) {
                float layer_vpad2 = nn_height / nn.as[l + 1].cols;

                for (size_t j = 0; j < nn.as[l + 1].cols; ++j) {
                    float cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
                    float cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;

                    // Color intensity based on weight strength
                    float value = sigmoidf(MAT_AT(nn.ws[l], i, j));
                    high_color.a = floorf(255.0f * value);
                    float thick = rh * 0.004f;

                    Vector2 start = {cx1, cy1};
                    Vector2 end = {cx2, cy2};
                    DrawLineEx(start, end, thick,
                              ColorAlphaBlend(low_color, high_color, WHITE));
                }
            }

            // Draw neuron
            if (l > 0) {
                // Color based on bias strength
                high_color.a = floorf(255.0f * sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)));
                DrawCircle(cx1, cy1, neuron_radius,
                          ColorAlphaBlend(low_color, high_color, WHITE));
            } else {
                // Input layer neurons are gray
                DrawCircle(cx1, cy1, neuron_radius, GRAY);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// gym_plot: Plot a sequence of values as a line graph
// -----------------------------------------------------------------------------
// This draws a line graph showing values over time, commonly used to
// visualize training cost.
//
void gym_plot(Plot plot, int rx, int ry, int rw, int rh)
{
    // Find min and max values for scaling
    float min = FLT_MAX, max = FLT_MIN;
    for (size_t i = 0; i < plot.count; ++i) {
        if (max < plot.items[i]) max = plot.items[i];
        if (min > plot.items[i]) min = plot.items[i];
    }
    if (min > 0) min = 0;  // Ensure graph starts at zero

    // Minimum width for the graph
    size_t n = plot.count;
    if (n < 1000) n = 1000;

    // Draw line segments connecting consecutive points
    for (size_t i = 0; i + 1 < plot.count; ++i) {
        float x1 = rx + (float)rw / n * i;
        float y1 = ry + (1 - (plot.items[i] - min) / (max - min)) * rh;
        float x2 = rx + (float)rw / n * (i + 1);
        float y2 = ry + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;
        DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh * 0.005f, RED);
    }

    // Draw zero line
    float y0 = ry + (1 - (0 - min) / (max - min)) * rh;
    DrawLineEx((Vector2){rx + 0, y0}, (Vector2){rx + rw - 1, y0},
              rh * 0.005f, WHITE);
    DrawText("0", rx + 0, y0 - rh * 0.04f, rh * 0.04f, WHITE);
}

// -----------------------------------------------------------------------------
// gym_process_batch: Process one batch of training data
// -----------------------------------------------------------------------------
// This function processes a mini-batch of training data, which is more
// efficient than processing one sample at a time and can help with
// convergence.
//
void gym_process_batch(Gym_Batch *gb, size_t batch_size, NN nn, NN g,
                       Mat t, float rate)
{
    // Reset if batch processing is complete
    if (gb->finished) {
        gb->finished = false;
        gb->begin = 0;
        gb->cost = 0;
    }

    // Calculate actual batch size (may be smaller at the end)
    size_t size = batch_size;
    if (gb->begin + batch_size >= t.rows) {
        size = t.rows - gb->begin;
    }

    // Create views for this batch's input and output data
    Mat batch_ti = {
        .rows = size,
        .cols = NN_INPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, gb->begin, 0),
    };

    Mat batch_to = {
        .rows = size,
        .cols = NN_OUTPUT(nn).cols,
        .stride = t.stride,
        .es = &MAT_AT(t, gb->begin, batch_ti.cols),
    };

    // Train on this batch
    nn_backprop(nn, g, batch_ti, batch_to);
    nn_learn(nn, g, rate);

    // Accumulate cost and advance to next batch
    gb->cost += nn_cost(nn, batch_ti, batch_to);
    gb->begin += batch_size;

    // Check if all batches are complete
    if (gb->begin >= t.rows) {
        size_t batch_count = (t.rows + batch_size - 1) / batch_size;
        gb->cost /= batch_count;  // Average cost across batches
        gb->finished = true;
    }
}

#endif // NN_ENABLE_GYM

#endif // NN_IMPLEMENTATION
