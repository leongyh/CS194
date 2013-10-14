__kernel void vvadd (__global float *Y, __global float *A, __global float *B, int n)
{
  /* CS194: implement the body of this kernel */
  int idx = get_global_id(0); //Get work item id at dim 0
  if(idx < n)
    {
      Y[idx] = A[idx] + B[idx];  //each work "thread" adds the arrays A and B at index and stores the sum to Y at index id.
    }
}
