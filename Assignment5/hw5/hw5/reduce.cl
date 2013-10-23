__kernel void reduce(__global int *in, __global int *out, __local int *buf, int n)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);

  //pulls data into local memory chunk
  buf[tid] = in[idx];
  barrier(CLK_LOCAL_MEM_FENCE);

  // reduce chunk locally in that work item. this should avoid bank conflicts.
  for(int offset = dim/2; offset > 0; offset >>= 1){
    if(tid < offset){
      int other = buf[tid + offset];
      int mine = buf[tid];
      buf[tid] = mine + other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // puts the reduced sum of the chunkback into the output array in
  // the index based on the work group ID.
  if(tid == 0){
    out[gid] = buf[0];
  }
}
