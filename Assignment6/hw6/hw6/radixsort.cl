/*
 the update kernel is modified to handle both
 zeros and ones in order to amortize overhead
 costs.
*/
__kernel void update(__global int *in_zeroes,
							 __global int *in_ones,
							 __global int *block_zeroes,
							 __global int *block_ones,
							 int n)
{
	size_t idx = get_global_id(0);
	size_t tid = get_local_id(0);
	size_t dim = get_local_size(0);
	size_t gid = get_group_id(0);

	if(idx < n && gid > 0)
	{
		in_zeroes[idx] = in_zeroes[idx] + block_zeroes[gid-1];
		in_ones[idx] = in_ones[idx] + block_ones[gid - 1];
	}
}

/*
the scan kernel is modified to handle both the zeros
and the ones in order to amortize overhead costs.
The same scan kernel of the sweep up, sweep down
technique is used from homework five.
*/
__kernel void scan(__global int *in_zeroes,
							 __global int *in_ones,
							 __global int *zeroes,
							 __global int *ones,  
							 __global int *bzeroes,
							 __global int *bones,
							 /* dynamically sized local (private) memory */
							 __local int *zeroes_buf,
							 __local int *ones_buf,
							 int v,
							 int k,
							 int n)
{
	size_t idx = get_global_id(0);
	size_t tid = get_local_id(0);
	size_t dim = get_local_size(0);
	size_t gid = get_group_id(0);
	int z, o = 0;

	if(idx<n){
		z = in_zeroes[idx];
		o = in_ones[idx];
		/* CS194: v==-1 used to signify 
		 * a "normal" additive scan
		 * used to update the partial scans */
		z = (v==-1) ? z : (v==((z>>k)&0x1)); 
		o = (v==-1) ? o : (v!=((o>>k)&0x1)); 

		zeroes_buf[tid] = z;
		ones_buf[tid] = o;
	}
	else{
		zeroes_buf[tid] = 0;
		ones_buf[tid] = 0;
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	//the first sweep up reduction step
	int offset = 1;
	for (int d = dim >> 1; d > 0; d >>= 1)
	{
		if(tid < d){
			zeroes_buf[offset * (2 * tid + 2) - 1] += zeroes_buf[offset * (2 * tid + 1) - 1];
			ones_buf[offset * (2 * tid + 2) - 1] += ones_buf[offset * (2 * tid + 1) - 1];
		}
		offset <<= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// at the last value in the local buffer to the other array for 
	//the update kernel so that this value can be added into the rest
	// of the values in the output array after this index.
	bzeroes[gid] = zeroes_buf[dim - 1];
	bones[gid] = ones_buf[dim - 1];

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// assigned last value in the calculated sweep up reduction step to 0
	zeroes_buf[dim - 1] = 0;
	ones_buf[dim - 1] = 0;

	// the final sweep down step
	for(int d = 1; d < dim; d <<= 1){
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if(tid < d){
			int a = offset * (2 * tid + 1) - 1;
			int b = offset * (2 * tid +  2) - 1;

			int temp0 = zeroes_buf[a];
			int temp1 = ones_buf[a];

			zeroes_buf[a] = zeroes_buf[b];
			ones_buf[a] = ones_buf[b];

			zeroes_buf[b] += temp0;
			ones_buf[b] += temp1;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// put the values back into the output array from the local buffer with
	// a one index shift. The last value the from the other array that was 
	// previously stored.
	if(idx < n){
		if(tid == dim - 1){
			zeroes[idx] = bzeroes[gid];
			ones[idx] = bones[gid];
		}else{
			zeroes[idx] = zeroes_buf[tid + 1];
			ones[idx] = ones_buf[tid + 1];
		}
	}
}

/*
 The reassemble kernel is used to rearrange the k-th bit
 based on the scanned zeros and ones. The zeros and ones
 array are loaded into local buffer so that they don't
 have to be globally retrieved. The speeds things up.
*/
__kernel void reassemble(__global int *in, 
								__global int *out, 
								__global int *zeroes, 
								__global int *ones, 
								__local int *temp_buf,
								__local int *zeroes_buf,
								__local int *ones_buf,
								int k, 
								int n){
	size_t idx = get_global_id(0);
	size_t tid = get_local_id(0);
	size_t dim = get_local_size(0);
	size_t gid = get_group_id(0);

	// those to local buffer
	int offset;
	if (idx < n){
		temp_buf[tid] = in[idx];
		zeroes_buf[tid] = zeroes[idx];
		ones_buf[tid] = ones[idx];
	}
	
	// reassemble
	if((temp_buf[tid] >> k) & 0x1){
		offset = zeroes[n - 1] + ones_buf[tid] - 1;
	}else{
		offset = zeroes_buf[tid] - 1;
	}
	out[offset] = temp_buf[tid];
}
