#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"

void recursive_reduce(cl_command_queue &queue,
        cl_context &context,
        cl_kernel &reduce_kern,
        cl_mem &in, 
        cl_mem &out, 
        int len){
  /*

  */

  size_t global_work_size[1] = {len};
  size_t local_work_size[1] = {512};

  if(len < 512){
    local_work_size[0] = len;
  }
  
  int left_over = 0;
  cl_int err;
  
  left_over = global_work_size[0] / local_work_size[0];

  err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &in);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 2, sizeof(int)*local_work_size[0], NULL);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 3, sizeof(int), &len);
  CHK_ERR(err);

  err = clEnqueueNDRangeKernel(cv.commands,
                  reduce,
                  1,
                  0,
                  global_work_size,
                  local_work_size,
                  0,
                  NULL,
                  NULL
                  );
  CHK_ERR(err);

  if(left_over > 1){
    recursive_reduce(queue,context,reduce_kern,out,out,left_over)
  }
}

int main(int argc, char *argv[])
{
  std::string reduce_kernel_str;
  
  std::string reduce_name_str = 
    std::string("reduce");
  std::string reduce_kernel_file = 
    std::string("reduce.cl");

  cl_vars_t cv; 
  cl_kernel reduce;
  
  readFile(reduce_kernel_file,
	   reduce_kernel_str);
  
  initialize_ocl(cv);
  
  compile_ocl_program(reduce, cv, reduce_kernel_str.c_str(),
		      reduce_name_str.c_str());

  int *h_A, *h_Y;
  cl_mem g_Out, g_In;
  int n = (1<<24);

  int c;
  /* how long do you want your arrays? */
  while((c = getopt(argc, argv, "n:"))!=-1){
    switch(c){
      case 'n':
        n = atoi(optarg);
        break;
    }
  }
  
  if(n==0)
    return 0;

  int padded_size = 1;
  
  while(padded_size < n){
    padded_size <<= 1;
  } 

  h_A = new int[padded_size];
  h_Y = new int[padded_size];

  for(int i = 0; i < n; i++){
    h_A[i] = 1;
    h_Y[i] = 0;
  }

  for (int i = n; i < padded_size; ++i)
  {
    h_A[i] = 0;
    h_Y[i] = 0;
  }

  cl_int err = CL_SUCCESS;
  g_Out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			 sizeof(int)*n,NULL,&err);
  CHK_ERR(err);  
  g_In = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			sizeof(int)*n,NULL,&err);
  CHK_ERR(err);

  //copy data from host CPU to GPU
  err = clEnqueueWriteBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n,
			     h_Y, 0, NULL, NULL);
  CHK_ERR(err);

  err = clEnqueueWriteBuffer(cv.commands, g_In, true, 0, sizeof(int)*n,
			     h_A, 0, NULL, NULL);
  CHK_ERR(err);
 
  size_t local_work_size[1] = {512};
  size_t global_work_size[1] = {padded_size};

  err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_In);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 2, sizeof(int)*512, NULL);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 3, sizeof(int), &padded_size);
  CHK_ERR(err);
  
  double t0 = timestamp();
  /* CS194 : Implement a reduction here */
  // for(int i = 0; i < 2; i++){
  // err = clEnqueueNDRangeKernel(cv.commands,
  //                 reduce,
  //                 1,
  //                 0,
  //                 global_work_size,
  //                 local_work_size,
  //                 0,
  //                 NULL,
  //                 NULL
  //                 );
  // CHK_ERR(err);
  // }
  recursive_reduce(cv.commands, reduce, g_In, g_Out, padded_size);
  t0 = timestamp()-t0;
  
  //read result of GPU on host CPU
  err = clEnqueueReadBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);
  
  int sum=0.0f;
  for(int i = 0; i < n; i++)
  {
    sum += h_A[i];
  }

  if(sum!=h_Y[0])
  {
    printf("WRONG: CPU sum = %d, GPU sum = %d\n", sum, h_Y[0]);
    printf("WRONG: difference = %d\n", sum-h_Y[0]);
    printf("Other parts = %d, %d, %d, %d\n", h_Y[1], h_Y[2], h_Y[3], h_Y[4]);
    int z=0;
    while(h_Y[z] == h_Y[z+1]){
	z++;
    }
    printf("red: %d\n", z);
  }
  else
  {
    printf("CORRECT: %d,%g\n",n,t0);
  }

  uninitialize_ocl(cv);
  
  delete [] h_A; 
  delete [] h_Y;
  
  clReleaseMemObject(g_Out); 
  clReleaseMemObject(g_In);
  
  return 0;
}
