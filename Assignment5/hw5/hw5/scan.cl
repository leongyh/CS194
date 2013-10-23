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
		   __local int *buf,
		   int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  buf[tid] = in[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  int offset = 1;
  for (int d = dim >> 1; d > 0; d >>= 1)
  {
    if(tid < d){
      buf[offset * (2 * tid + 2) - 1] += buf[offset * (2 * tid + 1) - 1];
    }
    offset <<= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  bout[gid] = buf[dim - 1];

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  buf[dim - 1] = 0;

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

  if(idx < n){
    if(tid == dim - 1){
      out[idx] = bout[gid];
    }else{
      out[idx] = buf[tid + 1];
    }
  }

/*
  buf[2 * tid] = in[2 * idx];
  buf[2 * tid + 1] = in[2 * idx + 1];
  barrier(CLK_LOCAL_MEM_FENCE);
  

  int offset = 1;
  for(int d = dim; d > 1; d >>= 1){
    if(tid < d){
      buf[offset * (2 * tid + 2) - 1] += buf[offset * (2 * tid + 1) - 1];
    }
    offset <<= 1;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  bout[gid] = buf[2 * dim - 1];

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  
  buf[2 * dim - 1] = 0;

  for(int d = 1; d < 2 * dim; d <<= 1){
    offset >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(tid < d){
      int a = offset * (2 * tid + 1) - 1;
      int b = offset * (2 * tid +  2) - 1;

      int temp = buf[a];
      buf[a] = buf[b];
      buf[b] = temp;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if(gid == 0){
    out[2 * idx] = buf[2 * tid];
    out[2 * idx + 1] = buf[2 * tid + 1];
  } else{
    out[2 * idx] = buf[2 * tid];
    out[2 * idx + 1] = buf[2 * tid + 1];
  }
*/
}
