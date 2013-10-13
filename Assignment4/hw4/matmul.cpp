#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>

#include "clhelp.h"

void sqr_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[])
{
  std::string matmul_kernel_str;
 
  std::string matmul_name_str = 
    std::string("matmul");
  std::string matmul_kernel_file = 
    std::string("matmul.cl");

  cl_vars_t cv; 
  cl_kernel matmul;
  
  readFile(matmul_kernel_file,
	   matmul_kernel_str);
  
  initialize_ocl(cv);
  
  compile_ocl_program(matmul, cv, matmul_kernel_str.c_str(),
		      matmul_name_str.c_str());
  
  float *h_A, *h_B, *h_Y, *h_YY;
  cl_mem g_A, g_B, g_Y;
  int n = (1<<10);
  h_A = new float[n*n];
  assert(h_A);
  h_B = new float[n*n];
  assert(h_B);
  h_Y = new float[n*n];
  assert(h_Y);
  h_YY = new float[n*n];
  assert(h_YY);
  bzero(h_Y, sizeof(float)*n*n);
  bzero(h_YY, sizeof(float)*n*n);
  
  for(int i = 0; i < (n*n); i++)
    {
      h_A[i] = (float)drand48();
      h_B[i] = (float)drand48();
    }


  cl_int err = CL_SUCCESS;
  /* CS194: Allocate Buffers on the GPU.
   *...We're already allocating the Y buffer
   * on the GPU for you */
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
		       sizeof(float)*n*n,NULL,&err);
  CHK_ERR(err);
  
  /* CS194: Copy data from host CPU to GPU */

  /* CS194: Create appropriately sized workgroups */
  size_t global_work_size[2] = {-1,-1};
  size_t local_work_size[2] = {-1,-1};
  
  /* CS194: Set kernel arguments */

  double t0 = timestamp();
  /* CS194: Launch matrix multiply kernel
   * Here's a little code to get you started.. 
   err = clEnqueueNDRangeKernel(cv.commands,...
			       );
   CHK_ERR(err);
   err = clFinish(cv.commands);
   CHK_ERR(err);
  */
  t0 = timestamp()-t0;


  /* Read result of GPU on host CPU */
  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);
  err = clFinish(cv.commands);
  CHK_ERR(err);

  double t1 = timestamp();
  sqr_sgemm(h_YY, h_A, h_B, n);
  t1 = timestamp()-t1;

  for(int i = 0; i < (n*n); i++)
    {
      double d = h_YY[i] - h_Y[i];
      d *= d;
      if(d > 0.0001)
	{
	  printf("CPU and GPU results do not match!\n");
	  break;
	}
    }
  uninitialize_ocl(cv);
  
  delete [] h_A; 
  delete [] h_B; 
  delete [] h_Y;
  delete [] h_YY;

  clReleaseMemObject(g_A); 
  clReleaseMemObject(g_B); 
  clReleaseMemObject(g_Y);
  
  double gpu_flops_s = (2.0 * pow((double)n, 3.0)) / t0;
  printf("GPU: %g gflops/sec\n", gpu_flops_s / (1e9));

  double cpu_flops_s = (2.0 * pow((double)n, 3.0)) / t1;
  printf("CPU: %g gflops/sec\n", cpu_flops_s / (1e9));
  return 0;
}
