#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cassert>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"


void cpu_scan(int *in, int *out, int n)
{
  out[0] = in[0];
  for(int i = 1; i < n; i++)
    out[i] = out[i-1]+in[i];
}

void recursive_scan(cl_command_queue &queue,
		    cl_context &context,
		    cl_kernel &scan_kern,
		    cl_kernel &update_kern,
		    cl_mem &in, 
		    cl_mem &out, 
		    int len)
{
  /* this is the upper-level recursion which calls the scan kernel
   that does work on 128 sized blocks. the kernel does a scan on 
   each individual block and uses the update kernel to update the blocks
   by adding the sum of the previous block's final prefix sum in the
   recursion. The update sum is always updated via recursion 
   until the end of the recursion and added finally back to the 
   output array.
  */
  size_t global_work_size[1] = {len};
  size_t local_work_size[1] = {128};
  int left_over = 0;
  cl_int err;
  
  adjustWorkSize(global_work_size[0], local_work_size[0]);
  global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);

  left_over = global_work_size[0] / local_work_size[0];
  
  cl_mem g_bscan = clCreateBuffer(context,CL_MEM_READ_WRITE, 
				  sizeof(int)*left_over,NULL,&err);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 0, sizeof(cl_mem), &in);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 1, sizeof(cl_mem), &out);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 2, sizeof(cl_mem), &g_bscan);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 3, 2*local_work_size[0]*sizeof(cl_int), NULL);
  CHK_ERR(err);

  err = clSetKernelArg(scan_kern, 4, sizeof(int), &len);
  CHK_ERR(err);

  /* explanation in the kernel file */
  err = clEnqueueNDRangeKernel(queue,
			       scan_kern,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);

  if(left_over > 1)
    {
      cl_mem g_bbscan = clCreateBuffer(context,CL_MEM_READ_WRITE, 
				      sizeof(int)*left_over,NULL,&err);

      recursive_scan(queue,context,scan_kern,update_kern,g_bscan,g_bbscan,left_over);


      err = clSetKernelArg(update_kern,0,
			   sizeof(cl_mem), &out);
      CHK_ERR(err);
      
      err = clSetKernelArg(update_kern,1,
			   sizeof(cl_mem), &g_bbscan);
      CHK_ERR(err);

      err = clSetKernelArg(update_kern,2,
			   sizeof(int), &len);
      CHK_ERR(err);
      
      /* 
      this update function is required as we are doing the scan
      in chunks rather than as a whole. The last value of the chunk
      will not be carried on to the next chunk. The update kernel
      is called in recursion to update all the values that missed out on
      the sum of the previous chunks' last values.
      */
      err = clEnqueueNDRangeKernel(queue,
				   update_kern,
				   1,//work_dim,
				   NULL, //global_work_offset
				   global_work_size, //global_work_size
				   local_work_size, //local_work_size
				   0, //num_events_in_wait_list
				   NULL, //event_wait_list
				   NULL //
				   );
      CHK_ERR(err);
      
      clReleaseMemObject(g_bbscan);
    }



  clReleaseMemObject(g_bscan);

}


int main(int argc, char *argv[])
{
  std::string kernel_source_str;
  
  std::string arraycompact_kernel_file = 
    std::string("scan.cl");
  
  std::list<std::string> kernel_names;
  std::string scan_name_str = std::string("scan");
  std::string update_name_str = std::string("update");

  kernel_names.push_back(scan_name_str);
  kernel_names.push_back(update_name_str);

  cl_vars_t cv; 
  
  std::map<std::string, cl_kernel> 
    kernel_map;

  int c;
  int n = (1<<20);  
  int *in, *out;

  int *cg_scan;
  
  while((c = getopt(argc, argv, "n:"))!=-1)
    {
      switch(c)
	{
	case 'n':
	  n = atoi(optarg);
	  break;
	}
    }

  in = new int[n];
  out = new int[n];
  cg_scan = new int[n];
  bzero(cg_scan, sizeof(int)*n);
  bzero(out, sizeof(int)*n);

  srand(5);
  for(int i = 0; i < n; i++)
    {
      in[i] = rand() %2;
    }

  /* Perform scan on CPU */
  cpu_scan(in,out,n);

  readFile(arraycompact_kernel_file,
	   kernel_source_str);
  
  initialize_ocl(cv);
  
  compile_ocl_program(kernel_map, cv, 
		      kernel_source_str.c_str(),
		      kernel_names);
  
  cl_mem g_in, g_scan;
  
  cl_int err = CL_SUCCESS;
  g_in = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
		       sizeof(int)*n,NULL,&err);
  CHK_ERR(err);  

  g_scan = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			  sizeof(int)*n,NULL,&err);
  CHK_ERR(err);
  
  
  //copy data from host CPU to GPU
  err = clEnqueueWriteBuffer(cv.commands, g_in, true, 0, sizeof(int)*n,
			     in, 0, NULL, NULL);
  CHK_ERR(err);

  
  double t0 = timestamp();

  size_t global_work_size[1] = {n};
  size_t local_work_size[1] = {128};

  adjustWorkSize(global_work_size[0], local_work_size[0]);

  global_work_size[0] = std::max(local_work_size[0], global_work_size[0]);
 
  /* perform scan */
  recursive_scan(cv.commands,cv.context,
		 kernel_map[scan_name_str],
		 kernel_map[update_name_str],
		 g_in,
		 g_scan,
		 n);

  err = clFlush(cv.commands);
  CHK_ERR(err);
  
  err = clEnqueueReadBuffer(cv.commands, g_scan, true, 0, sizeof(int)*n,
			    cg_scan, 0, NULL, NULL);
  CHK_ERR(err);
  
  /* CHECK CORRECTNESS */
  for(int i =0; i < n; i++)
    {
      if(out[i] != cg_scan[i])
	{
	  printf("scan mismatch @ %d: cpu=%d, gpu=%d\n", 
		 i, out[i], cg_scan[i]);
	  return 1;
	}
    }
  
  printf("%s\n", "success");

  clReleaseMemObject(g_in); 
  clReleaseMemObject(g_scan);

  uninitialize_ocl(cv);

  delete [] in;
  delete [] out;
  delete [] cg_scan;

  return 0;
}
