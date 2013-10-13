#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "clhelp.h"

int main(int argc, char *argv[])
{
  std::string incr_kernel_str;

  /* Provide names of the OpenCL kernels
   * and cl file that they're kept in */
  std::string incr_name_str = 
    std::string("incr");
  std::string incr_kernel_file = 
    std::string("incr.cl");

  cl_vars_t cv; 
  cl_kernel incr;

  /* Read OpenCL file into STL string */
  readFile(incr_kernel_file,
	   incr_kernel_str);
  
  /* Initialize the OpenCL runtime 
   * Source in clhelp.cpp */
  initialize_ocl(cv);
  
  /* Compile all OpenCL kernels */
  compile_ocl_program(incr, cv, incr_kernel_str.c_str(),
		      incr_name_str.c_str());
  
  /* Arrays on the host (CPU) */
  float *h_Y, *h_YY;
  /* Arrays on the device (GPU) */
  cl_mem g_Y;

  int n = (1<<20);
  h_Y = new float[n];
  h_YY = new float[n];
   
  for(int i = 0; i < n; i++)
    {
      h_YY[i] = h_Y[i] = (float)drand48();
    }

  cl_int err = CL_SUCCESS;
  /* CS194: Allocate memory for arrays on 
   * the GPU */
  g_Y = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,sizeof(float)*n,NULL,&err);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n,
			     h_Y, 0, NULL, NULL);
  CHK_ERR(err);

  size_t global_work_size[1] = {n};
  size_t local_work_size[1] = {128};
    
  err = clSetKernelArg(incr, 0, sizeof(cl_mem), &g_Y);
  CHK_ERR(err);
  err = clSetKernelArg(incr, 1, sizeof(int), &n);
  CHK_ERR(err);
 
  err = clEnqueueNDRangeKernel(cv.commands,
			       incr,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);

  /* Read result of GPU on host CPU */
  err = clEnqueueReadBuffer(cv.commands, g_Y, true, 0, sizeof(float)*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);

  /* Check answer */
  bool er = false;
  for(int i = 0; i < n; i++)
    {
      float d = (h_YY[i] + 1.0f);
      if(h_Y[i] != d)
	{
	  printf("error at %d :(\n", i);
	  er = true;
	  break;
	}
    }
  if(!er)
    {
      printf("CPU and GPU results match\n");
    }

  uninitialize_ocl(cv);
  
  delete [] h_Y;
  delete [] h_YY;

  clReleaseMemObject(g_Y);
  
  return 0;
}
