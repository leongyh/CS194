#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "clhelp.h"

int main(int argc, char *argv[])
{
  std::string vvadd_kernel_str;

  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in */
  std::string vvadd_name_str = 
    std::string("vvadd");
  std::string vvadd_kernel_file = 
    std::string("vvadd.cl");

  cl_vars_t cv; 
  cl_kernel vvadd;

  /* Read OpenCL file into STL string */
  readFile(vvadd_kernel_file,
	   vvadd_kernel_str);
  
  /* Initialize the OpenCL runtime 
   * Source in clhelp.cpp */
  initialize_ocl(cv);
  
  /* Compile all OpenCL kernels */
  compile_ocl_program(vvadd, cv, vvadd_kernel_str.c_str(),
		      vvadd_name_str.c_str());
  
  /* Arrays on the host (CPU) */
  float *h_A, *h_B, *h_Y;
  /* Arrays on the device (GPU) */
  cl_mem g_A, g_B, g_Y;

  /* Allocate arrays on the host
   * and fill with random data */
  int n = (1<<20);
  h_A = new float[n];
  h_B = new float[n];
  h_Y = new float[n];
  bzero(h_Y, sizeof(float)*n);
  
  for(int i = 0; i < n; i++)
    {
      h_A[i] = (float)drand48();
      h_B[i] = (float)drand48();
    }

  /* CS194: Allocate memory for arrays on 
   * the GPU */
  cl_int err = CL_SUCCESS;
  
  /* CS194: Here's something to get you started  */
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err);
  CHK_ERR(err);
  

  /* CS194: Copy data from host CPU to GPU */
 
  /* CS194: Define the global and local workgroup sizes */
  size_t global_work_size[1] = {0};
  size_t local_work_size[1] = {0};
  
  /* CS194: Set Kernel Arguments */

  /* CS194: Call kernel on the GPU */

  /* Read result of GPU on host CPU */
  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);

  /* Check answer */
  for(int i = 0; i < n; i++)
    {
      float d = h_A[i] + h_B[i];
      if(h_Y[i] != d)
	{
	  printf("error at %d :(\n", i);
	  break;
	}
    }

  /* Shut down the OpenCL runtime */
  uninitialize_ocl(cv);
  
  delete [] h_A; 
  delete [] h_B; 
  delete [] h_Y;
  
  clReleaseMemObject(g_A); 
  clReleaseMemObject(g_B); 
  clReleaseMemObject(g_Y);
  
  return 0;
}
