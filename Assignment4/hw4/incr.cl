__kernel void incr (__global float *Y, int n)
{
  int idx = get_global_id(0);
  if(idx < n)
    {
      Y[idx] = Y[idx] + 1.0f;
    }
}
