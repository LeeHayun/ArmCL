#include "helpers.h"

/** This kernel computes the matrix by vector multiplication
 *
 * @note This OpenCL kernel works with floating point data types (F32)
 *
 * @param[in]  src0_ptr
 * @param[in]  src0_stride_x
 * @param[in]  src0_step_x
 * @param[in]  src0_offset_first_element_in_bytes
 * @param[in]  src1_val_ptr
 * @param[in]  src1_val_stride_x
 * @param[in]  src1_val_step_x
 * @param[in]  src1_val_offset_first_element_in_bytes
 * @param[in]  src1_row_ptr_ptr
 * @param[in]  src1_row_ptr_stride_x
 * @param[in]  src1_row_ptr_step_x
 * @param[in]  src1_row_ptr_offset_first_element_in_bytes
 * @param[in]  src1_col_idx_ptr
 * @param[in]  src1_col_idx_stride_x
 * @param[in]  src1_col_idx_step_x
 * @param[in]  src1_col_idx_offset_first_element_in_bytes
 * @param[in]  dst_ptr
 * @param[in]  dst_stride_x
 * @param[in]  dst_step_x
 * @param[in]  dst_offset_first_element_in_bytes
 */
__kernel void sparse_csr_gemv(VECTOR_DECLARATION(src0),
                              VECTOR_DECLARATION(src1_val),
                              VECTOR_DECLARATION(src1_row_ptr),
                              VECTOR_DECLARATION(src1_col_idx),
                              VECTOR_DECLARATION(dst))
{
    __private int c;
    __private int i = get_global_id(0);

    __global float *src0_addr = (__global float *)(src0_ptr + src0_offset_first_element_in_bytes);
    __global float *src1_val_addr = (__global float *)(src1_val_ptr + src1_val_offset_first_element_in_bytes);
    __global int *src1_row_ptr_addr = (global float *)(src1_row_ptr_ptr + src1_row_ptr_offset_first_element_in_bytes);
    __global int *src1_col_idx_addr = (global float *)(src1_col_idx_ptr + src1_col_idx_offset_first_element_in_bytes);

    __private const int rowstart = src1_row_ptr_addr[i];
    __private const int rowend = src1_row_ptr_addr[i+1];

    float acumulador=0;
    float4 auxV;
    float4 auxAA;
    int4 kk;

    for( c = rowstart; c < rowend; c+=4)
    {

        //auxAA.x = src1_val_addr[c];
        //auxAA.y = src1_val_addr[c + 1];
        //auxAA.z = src1_val_addr[c + 2];
        //auxAA.w = src1_val_addr[c + 3];
        auxAA = vload4(0, src1_val_addr + c);
        kk = ((int4)(c, c + 1, c + 2, c + 3) <= (rowend - 1));
        auxV.x = kk.x ? src0_addr[src1_col_idx_addr[c]] : 0;
        auxV.y = kk.y ? src0_addr[src1_col_idx_addr[c + 1]] : 0;
        auxV.z = kk.z ? src0_addr[src1_col_idx_addr[c + 2]] : 0;
        auxV.w = kk.w ? src0_addr[src1_col_idx_addr[c + 3]] : 0;
        acumulador += dot(auxAA, auxV);

        //dst_ptr[i] += src1_val_addr[c] * src0_ptr[src1_col_idx_ptr[c]];

    }

    *((__global float *)(dst_ptr + dst_offset_first_element_in_bytes) + i) = acumulador;
}
