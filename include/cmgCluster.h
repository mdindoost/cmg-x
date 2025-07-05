#ifndef CMG_CLUSTER_H
#define CMG_CLUSTER_H

#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

// Sparse matrix structure matching scipy's CSC format
typedef struct {
    double* data;      // Nonzero values
    size_t* indices;   // Row indices
    size_t* indptr;    // Column pointers
    size_t n_rows;     // Number of rows
    size_t n_cols;     // Number of columns
    size_t nnz;        // Number of nonzero elements
} csc_matrix;

// Internal helper functions
void split_forest(uint32_t* C, size_t n);


static uint32_t forest_components(
    const uint32_t* C,    // input forest
    size_t n,             // size of arrays
    uint32_t* cI,         // component indices output
    uint32_t* csizes      // component sizes output (optional, can be NULL)
);

static void update_groups_sparse(
    const double* values,     // non-zero values
    const size_t* row_ind,    // row indices
    const size_t* col_ptr,    // column pointers
    size_t n,                 // matrix dimension
    size_t nnz,               // number of non-zeros
    const uint32_t* C,        // input grouping
    const double* dA,         // diagonal entries
    uint32_t* C1              // output grouping
);

// Main exported function
EXPORT uint32_t steiner_group(
    const csc_matrix* A,    // input Laplacian matrix
    uint32_t* cI            // output component indices (0-based)
);

#endif