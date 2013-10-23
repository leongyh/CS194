__kernel void reduce(__global int *in, __global int *out, __local int *buf, int n)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);

  /*
  int acc = 0;
  while(idx < n){
    int elem = in[idx];
    acc += elem;
    idx += get_global_size(0);
  }

  buf[tid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  */

  buf[tid] = in[idx];
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for(int offset = dim/2; offset > 0; offset >>= 1){
    if(tid < offset){
      int other = buf[tid + offset];
      int mine = buf[tid];
      buf[tid] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(tid == 0){
    out[gid] = buf[0];
  }

/*
  int i = gid*(dim*2) + tid;

  buf[tid] = (i < n) ? in[i] : 0;

  if(i + dim < n)
    buf[tid] += in[i + dim];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = dim/2; i > 0; i >>= 1)
  {
    if(tid < i)
    {
      buf[tid] += buf[tid + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(tid == 0)
    out[gid] = buf[0];
  */
}
