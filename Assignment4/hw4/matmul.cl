__kernel void matmul(__global float *Y, __global float *A, __global float *B, int n)
{
  /* CS194: Implement the body of this kernel */
  int idx = get_global_id(0); //Get work item id at dim 0
  int idy = get_global_id(1); //Get work item id at dim 1
  float tmp = 0;

  if(idx < n && idy < n)
  {
	for(int k = 0; k < n; k++)
	{
	  tmp += A[idx*n + k] * B[k*n + idy];
	}

	Y[idx*n + idy] = tmp;
  }
}
