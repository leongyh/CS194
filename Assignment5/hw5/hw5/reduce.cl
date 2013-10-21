__kernel void reduce(__global int *in, __global int *out, __local int *buf, int n)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);

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
}
