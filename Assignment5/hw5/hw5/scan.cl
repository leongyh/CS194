__kernel void update(__global int *in,
		     __global int *block,
		     int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if(idx < n && gid > 0)
    {
      in[idx] = in[idx] + block[gid-1];
    }
}

/* the kernel uses Blelloch's implementation (http://
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.128.6230&
rep=rep1&type=pdf) of this sweep up and sweep down the method
to calculate the prefix sum block by block. the final  prefix sum
of each block is saved to another array so that it can be used for
the update kernel to add it back into the final output array during
the upper-level recursion.
*/
__kernel void scan(__global int *in, 
		   __global int *out, 
		   __global int *bout,
		   __local int *buf,
		   int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  //add a local sized chuck to local buffer
  buf[tid] = in[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  //the first sweep up reduction step
  int offset = 1;
  for (int d = dim >> 1; d > 0; d >>= 1)
  {
    if(tid < d){
      buf[offset * (2 * tid + 2) - 1] += buf[offset * (2 * tid + 1) - 1];
    }
    offset <<= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // at the last value in the local buffer to the other array for 
  //the update kernel so that this value can be added into the rest
  // of the values in the output array after this index.
  bout[gid] = buf[dim - 1];

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // assigned last value in the calculated sweep up reduction step to 0
  buf[dim - 1] = 0;

  // the final sweep down step
  for(int d = 1; d < dim; d <<= 1){
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(tid < d){
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid +  2) - 1;

      int temp = buf[a];
      buf[a] = buf[b];
      buf[b] += temp;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // put the values back into the output array from the local buffer with
  // a one index shift. The last value the from the other array that was 
  // previously stored.
  if(idx < n){
    if(tid == dim - 1){
      out[idx] = bout[gid];
    }else{
      out[idx] = buf[tid + 1];
    }
  }
}
