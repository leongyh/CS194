/* The kernel of the incr function increases each array value by 1.0*/
__kernel void incr (__global float *Y, int n)
{
  int idx = get_global_id(0); //Get work item id at dim 0
  if(idx < n)
    {
      Y[idx] = Y[idx] + 1.0f;  //each work "thread" increments the array at index id by 1.0
    }
}