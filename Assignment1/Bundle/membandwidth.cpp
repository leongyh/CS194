#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include "simd_copy.cpp"

int main(int argc, char *argv[])
{
	struct timeval tv;

	//Get the number of the user. number = bytes of the array.
	size_t size = atoi(argv[1]);

	//warmup cache
	cacheWarmup(size, 100);

	//naive copy
	int* source_array = new source_array[size];
	for (int i = 0; i < (int) size/4; ++i)
	{
		source_array[i] = i;
	}
	int* naive_array = new naive_array[size];

	gettimeofday(&tv, 0);
	double time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	memcpy(naive_array, source_array, sizeof(source_array));

	double time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for naive copy: %f seconds.\n", time_taken);

	//simd_memcpy
	int* source_array = new source_array[size];
	for (int i = 0; i < (int) size/4; ++i)
	{
		source_array[i] = i;
	}
	int* simd_array = new simd_array[size];

	gettimeofday(&tv, 0);
	double time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	simd_memcpy(simd_array, source_array, sizeof(source_array));

	double time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for simd copy: %f seconds.\n", time_taken);

	//simd_memcpy_cache
	int* source_array = new source_array[size];
	for (int i = 0; i < (int) size/4; ++i)
	{
		source_array[i] = i;
	}
	int* simd_cache_array = new simd_cache_array[size];

	gettimeofday(&tv, 0);
	double time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	memcpy(simd_cache_array, source_array, sizeof(source_array));

	double time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for simd cache copy: %f seconds.\n", time_taken);
}

void cacheWarmup(size_t size, int iter)
{
	//create and populate array (32-bit, 4 byte slots)
	int* arr = new arr[size];

	for (int i = 0; i < (int) size/4; ++i)
	{
		arr[i] = i;
	}
	
	//warm up cache
	int warmup_count = iter;

	for (int i = 0; i < warmup_count; ++i)
	{
		int* temp_arr = new arr[size];
		memcpy(temp_arr, arr, sizeof(arr));
	}
}