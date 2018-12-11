__kernel void sparse_csr_gemm(__global int *row_pointer,
                            __global int *col_index,
                            __global double *value,
                            __global double *input_matrix,
                            __global double *result
                            )

{
    int c, k;
    int i = get_global_id(0);
    int col_idx;
    double csr_value;
    int csr_row = 0;

    for( c = row_pointer[i]; c < row_pointer[i+1]; c++)
    {
        col_idx = col_index[c];
        csr_value = value[c];

        for ( k = 0; k < i; k++)
            {
                result[k][col_idx] += csr_value * input_matrix[k][csr_row];
            }

    }

    csr_row += 1;

}

