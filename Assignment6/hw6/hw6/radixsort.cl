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

__kernel void scan(__global int *in, 
		   __global int *out, 
		   __global int *bout,
		   /* dynamically sized local (private) memory */
		   __local int *buf, 
		   int v,
		   int k,
		   int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);
  int t, r = 0, w = dim;

  if(idx<n){
    t = in[idx];
    /* CS194: v==-1 used to signify 
     * a "normal" additive scan
     * used to update the partial scans */
    t = (v==-1) ? t : (v==((t>>k)&0x1)); 
    buf[tid] = t;
  }
  else{
    buf[tid] = 0;
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

  /* CS194: Your scan code from HW 5 goes here */

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

  /* CS194: Store partial scans */
  /*
  if(idx < n)
    {
      out[idx] = buf[r+tid];
    }
  */
  /* CS194: one work-item stores the
   * work group's total partial
   * "reduction" */
  /*
  if(tid==0)
    {
      bout[gid] = buf[r+dim-1];
    }
  */
}

__kernel void reassemble(__global int *in, __global int *out, __global int *zeros, __global int *ones, __local int *temp_buf,/* __local int *zeros_buf, __local int *ones_buf,*/ int k, int n){
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  int offset;
  if (idx < n){
    temp_buf[tid] = ((in[idx] >> k) & 0x1);
    //zeros_buf[tid] = zeros[idx];
    //ones_buf[tid] = ones[idx];
  }
  
  if(temp_buf[tid]){
    offset = zeros[n - 1] + ones[idx] - 1;
  }else{
    offset = zeros[idx] - 1;
  }
  out[offset] = in[idx];
  barrier(CLK_GLOBAL_MEM_FENCE);
}
