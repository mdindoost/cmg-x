#include "cmgCluster.h"
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/* forest_components(): FIXED to match MATLAB forest_components_ */
uint32_t forest_components(const uint32_t* C, size_t n, uint32_t* cI, uint32_t* csizes) {
    //printf("[C] forest_components() called\n");
    fflush(stdout);

    memset(cI, 0, n * sizeof(uint32_t));
    if (csizes != NULL) memset(csizes, 0, n * sizeof(uint32_t));

    // Dynamic buffer management like MATLAB
    size_t buffer_size = 100;
    uint32_t* buffer = malloc(buffer_size * sizeof(uint32_t));
    if (!buffer) return 0;

    uint32_t ccI = 1;
    for (size_t j = 0; j < n; j++) {
        if (cI[j] > 0) continue;

        size_t bufferI = 0;  // MATLAB uses 1-based, C uses 0-based
        size_t jwalk = j;

        // Tree walk until we hit a labeled node
        while (cI[jwalk] == 0) {
            cI[jwalk] = ccI;
            buffer[bufferI] = jwalk;
            bufferI++;

            // Dynamic buffer expansion like MATLAB
            if (bufferI >= buffer_size) {
                size_t new_size = (buffer_size * 2 < n) ? buffer_size * 2 : n;
                uint32_t* new_buffer = realloc(buffer, new_size * sizeof(uint32_t));
                if (!new_buffer) {
                    free(buffer);
                    return 0;
                }
                buffer = new_buffer;
                buffer_size = new_size;
            }

            jwalk = C[jwalk];
        }

        // MATLAB: en = C(jwalk); but jwalk is already the end node
        // The walk stopped at jwalk because cI[jwalk] != 0
        uint32_t en = jwalk;  // The end node is jwalk itself
        
        if (cI[en] != ccI) {
            // Relabel all nodes in buffer to the existing component
            for (size_t i = 0; i < bufferI; i++) {
                cI[buffer[i]] = cI[en];
            }
        } else {
            // This is a new component
            ccI++;
        }
        
        // MATLAB: csizes(en) = csizes(en) + bufferI;
        if (csizes != NULL) {
            csizes[en] += bufferI;
        }
    }

    // MATLAB logic: ccI tracks "next available ID", so final count is ccI-1
    // The actual number of components used is ccI-1
    uint32_t total = (ccI > 1) ? ccI - 1 : 1;

    free(buffer);

    // printf("[C] forest_components() completed, found %u components\n", total);
    // printf("[C] Component labels (1-based):\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  cI[%zu] = %u\n", i, cI[i]);
    // }
    
    return total;
}

/* split_forest(): CORRECTED to exactly match MATLAB split_forest_ */
void split_forest(uint32_t* C, size_t n) {
    // printf("[C] split_forest() called with n = %zu\n", n);
    fflush(stdout);

    uint32_t* ancestors = calloc(n, sizeof(uint32_t));
    uint32_t* indegree = calloc(n + 2, sizeof(uint32_t));  // +2 like MATLAB
    uint8_t* visited = calloc(n, sizeof(uint8_t));
    
    const size_t WALK_BUFFER_SIZE = 20;  // Like MATLAB
    uint32_t walkbuffer[WALK_BUFFER_SIZE];
    uint32_t newancestorbuff[WALK_BUFFER_SIZE];

    if (!ancestors || !indegree || !visited) {
        free(ancestors); free(indegree); free(visited);
        return;
    }

    // printf("[C] Initial forest C:\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  C[%zu] = %u\n", i, C[i]);
    // }

    // Step 1: Compute indegrees (MATLAB: for j=1:n; indegree(C(j)) = indegree(C(j))+1; end)
    for (size_t j = 0; j < n; j++) {
        indegree[C[j]]++;
    }
    // printf("[C] Computed indegree vector\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  indegree[%zu] = %u\n", i, indegree[i]);
    // }

    // Step 2: Prune long paths (cluster diameter > 6)
    // printf("[C] Step 2: Pruning long paths\n");
    for (size_t j = 0; j < n; j++) {
        size_t jwalk = j;
        uint8_t startwalk = 1;

        while (startwalk && indegree[jwalk] == 0 && !visited[jwalk]) {
            startwalk = 0;
            uint32_t ancestors_in_path = 0;
            size_t k = 1;  // MATLAB uses 1-based indexing here
            walkbuffer[k-1] = jwalk;  // Convert to 0-based for C array
            newancestorbuff[k-1] = 0;
            
            // printf("[C] 🌲 Start walk from node %zu\n", jwalk);

            while (k <= 6 && !visited[jwalk]) {
                jwalk = C[jwalk];
                
                // MATLAB: if jwalk == walkbuffer(k) || (k > 1 && jwalk == walkbuffer(k-1))
                uint8_t cycle_detected = (jwalk == walkbuffer[k-1]) || 
                                        (k > 1 && jwalk == walkbuffer[k-2]);
                if (cycle_detected) {
                    // printf("[C] 🚫 Cycle or backtrack detected. Breaking at node %zu\n", jwalk);
                    break;
                }

                k++;
                if (k > WALK_BUFFER_SIZE) break;  // Safety check
                
                walkbuffer[k-1] = jwalk;  // Convert to 0-based
                
                if (visited[jwalk]) {
                    newancestorbuff[k-1] = ancestors_in_path;
                } else {
                    ancestors_in_path++;
                    newancestorbuff[k-1] = ancestors_in_path;
                }
                
                // printf("[C]   Walking... k=%zu, jwalk=%zu, ancestors_in_path=%u\n", 
                    //    k, jwalk, ancestors_in_path);
            }

            // MATLAB: if (k > 6)
            if (k > 6) {
                size_t middlek = (k + 1) / 2;  // MATLAB: ceil(k/2), convert to 0-based
                // printf("[C] ✂️ Long path detected (k=%zu). Cutting at middle node %u\n", 
                //        k, walkbuffer[middlek-1]);
                
                // MATLAB: C(walkbuffer(middlek)) = walkbuffer(middlek)
                C[walkbuffer[middlek-1]] = walkbuffer[middlek-1];  // Self-loop cut
                
                // MATLAB: indegree(walkbuffer(middlek+1)) = indegree(walkbuffer(middlek+1)) - 1
                if (middlek < k) {
                    indegree[walkbuffer[middlek]]--;
                }
                
                // MATLAB: for ik = middlek+1:k; ancestors(walkbuffer(ik)) = ancestors(walkbuffer(ik)) - ancestors(walkbuffer(middlek)); end
                for (size_t ik = middlek; ik < k; ik++) {
                    ancestors[walkbuffer[ik]] -= ancestors[walkbuffer[middlek-1]];
                }
                
                // MATLAB: for ik = 1:middlek; visited(walkbuffer(ik)) = true; ancestors(walkbuffer(ik)) = ancestors(walkbuffer(ik)) + newancestorbuff(ik); end
                for (size_t ik = 0; ik < middlek; ik++) {
                    visited[walkbuffer[ik]] = 1;
                    ancestors[walkbuffer[ik]] += newancestorbuff[ik];
                }
                
                // MATLAB: jwalk = walkbuffer(middlek+1); startwalk = true;
                if (middlek < k) {
                    jwalk = walkbuffer[middlek];
                    startwalk = 1;
                }
            } else {
                // MATLAB: if ~startwalk; for ik = 1:k; visited(walkbuffer(ik)) = true; ancestors(walkbuffer(ik)) = ancestors(walkbuffer(ik)) + newancestorbuff(ik); end; end
                for (size_t ik = 0; ik < k; ik++) {
                    visited[walkbuffer[ik]] = 1;
                    ancestors[walkbuffer[ik]] += newancestorbuff[ik];
                }
            }
        }
    }

    // printf("[C] Ancestors after Step 2:\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  ancestors[%zu] = %u\n", i, ancestors[i]);
    // }

    // Step 3: Remove low-conductance branches
    // printf("[C] Step 3: Removing low-conductance branches\n");
    for (size_t j = 0; j < n; j++) {
        size_t jwalk = j;
        uint8_t startwalk = 1;

        while (startwalk && indegree[jwalk] == 0) {
            startwalk = 0;
            size_t jwalkb = jwalk;
            uint8_t cut_mode = 0;
            uint32_t removed_ancestors = 0;
            size_t new_front = 0;

            while (1) {
                size_t jwalka = C[jwalk];
                
                // MATLAB: if jwalka == jwalk || jwalka == jwalkb; break; end
                if (jwalka == jwalk || jwalka == jwalkb) break;

                // MATLAB: if (~cut_mode && ancestors(jwalk) > 2 && (ancestors(jwalka) - ancestors(jwalk)) > 2)
                if (!cut_mode && ancestors[jwalk] > 2 && 
                    (ancestors[jwalka] - ancestors[jwalk]) > 2) {
                    
                    // printf("[C] 🔪 Cutting low-conductance edge at node %zu -> %zu\n", jwalk, jwalka);
                    // printf("[C]   ancestors[%zu]=%u, ancestors[%zu]=%u, diff=%u\n", 
                    //        jwalk, ancestors[jwalk], jwalka, ancestors[jwalka], 
                    //        ancestors[jwalka] - ancestors[jwalk]);
                    
                    // MATLAB: C(jwalk) = jwalk
                    C[jwalk] = jwalk;  // Break the link
                    // MATLAB: indegree(jwalka) = indegree(jwalka) - 1
                    indegree[jwalka]--;
                    removed_ancestors = ancestors[jwalk];
                    new_front = jwalka;
                    cut_mode = 1;
                }

                jwalkb = jwalk;
                jwalk = jwalka;
                
                // MATLAB: if cut_mode; ancestors(jwalk) = ancestors(jwalk) - removed_ancestors; end
                if (cut_mode) {
                    ancestors[jwalk] -= removed_ancestors;
                }
            }

            // MATLAB: if cut_mode; startwalk = true; jwalk = new_front; end
            if (cut_mode) {
                startwalk = 1;
                jwalk = new_front;
            }
        }
    }

    // printf("[C] Final forest after split_forest:\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  C[%zu] = %u\n", i, C[i]);
    // }
    // printf("[C] ✅ Finished forest splitting\n");

    free(ancestors);
    free(indegree);
    free(visited);
}

/* update_groups_sparse(): FIXED to match MATLAB update_groups_ */
static void update_groups_sparse(
    const double* values,
    const size_t* row_ind,
    const size_t* col_ptr,
    size_t n,
    size_t nnz,
    const uint32_t* C,
    const double* dA,
    uint32_t* C1
) {
    // printf("[C] update_groups_sparse() called\n");
    fflush(stdout);

    double* B = calloc(n, sizeof(double));
    if (!B) return;

    memcpy(C1, C, n * sizeof(uint32_t));

    // MATLAB: B(j) is the total tree weight incident to node j
    for (size_t j = 0; j < n; j++) {
        size_t k = C[j];
        if (k != j) {
            double a = 0.0;
            // Find A(j,k) in the sparse matrix
            for (size_t p = col_ptr[k]; p < col_ptr[k + 1]; p++) {
                if (row_ind[p] == j) {
                    a = values[p];
                    break;
                }
            }
            B[j] += a;
            B[k] += a;
        }
    }

    // MATLAB: ndx = find((B./dA_)>-0.125); C(ndx) = uint32(ndx);
    // Note: negative sign because A is Laplacian
    for (size_t j = 0; j < n; j++) {
        if ((B[j] / dA[j]) > -0.125) {
            C1[j] = j;  // Self-loop (remove from tree)
        }
    }

    free(B);
    // printf("[C] update_groups_sparse() completed\n");
    fflush(stdout);
}

/* steiner_group(): FIXED to match MATLAB steiner_group algorithm */
uint32_t steiner_group(const csc_matrix* A, uint32_t* cI) {
    // printf("======== STEINER GROUP BEGIN ========\n");
    // printf("Matrix: n_rows=%zu, n_cols=%zu, nnz=%zu\n", A->n_rows, A->n_cols, A->nnz);

    size_t n = A->n_rows;
    double* M = malloc(n * sizeof(double));
    uint32_t* C1 = malloc(n * sizeof(uint32_t));
    uint32_t* C = malloc(n * sizeof(uint32_t));
    double* dA = malloc(n * sizeof(double));
    double* efd = malloc(n * sizeof(double));
    uint32_t* csizes = malloc(n * sizeof(uint32_t));

    if (!M || !C1 || !C || !dA || !efd || !csizes) {
        // printf("❌ Memory allocation failed\n");
        goto cleanup;
    }

    // Step 1: MATLAB [M, C1] = min(A) - column-wise minimum
    for (size_t j = 0; j < n; j++) {
        M[j] = DBL_MAX;
        C1[j] = j;  // fallback to self
        dA[j] = 0.0;  // will be set when we find diagonal
        
        // Find minimum in column j and diagonal element
        for (size_t p = A->indptr[j]; p < A->indptr[j + 1]; p++) {
            size_t i = A->indices[p];
            double val = A->data[p];
            
            if (i == j) {
                dA[j] = val;  // Diagonal element
            } else {
                // For Laplacian, off-diagonal elements are negative
                // We want the minimum (most negative) value
                if (val < M[j]) {
                    M[j] = val;
                    C1[j] = i;
                }
            }
        }
        
        // // Convert 1-based MATLAB indexing to 0-based C indexing is already done
        // printf("  Column %zu: min_val=%.6f, min_idx=%u, diag=%.6f\n", 
        //        j, M[j], C1[j], dA[j]);
    }

    // Step 2: Apply forest decomposition
    // printf("\n[C] Calling split_forest()\n");
    memcpy(C, C1, n * sizeof(uint32_t));
    split_forest(C, n);

    // Step 3: Compute effective degrees
    // printf("\n[C] Computing effective degrees\n");
    double min_efd = DBL_MAX;
    for (size_t j = 0; j < n; j++) {
        // MATLAB: efd = abs(M'./dA_);
        efd[j] = fabs(M[j] / dA[j]);
        // printf("  efd[%zu] = %.6f\n", j, efd[j]);
        if (efd[j] < min_efd) {
            min_efd = efd[j];
        }
    }

    // Step 4: MATLAB: if min(efd) < (1/8)
    if (min_efd < 0.125) {  // 1/8 = 0.125
        // printf("\n[C] Applying update_groups_sparse (min_efd = %.6f < 0.125)\n", min_efd);
        update_groups_sparse(A->data, A->indices, A->indptr, n, A->nnz, C, dA, C1);
        memcpy(C, C1, n * sizeof(uint32_t));
    }

    // Step 5: Connected component labeling
    // printf("\n[C] Calling forest_components()\n");
    uint32_t nc = forest_components(C, n, cI, csizes);

    // printf("\n[C] Final component labels:\n");
    // for (size_t i = 0; i < n; i++) {
    //     printf("  label[%zu] = %u\n", i, cI[i]);
    // }

    // printf("[C] Total components: %u\n", nc);

cleanup:
    free(M); free(C1); free(C); free(dA); free(efd); free(csizes);
    // printf("======== STEINER GROUP END =========\n\n");
    fflush(stdout);
    return nc;
}