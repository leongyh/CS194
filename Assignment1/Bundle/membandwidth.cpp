#include <emmintrin.h>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sys/time.h>
#include <time.h>
#include <cstdio>
#include <iostream>

using namespace std;

void cacheWarmup(int size, int iter);
void simd_memcpy(void *dst, void *src, size_t nbytes);
void simd_memcpy_cache(void *dst, void *src, size_t nbytes);

int main(int argc, char *argv[])
{
	struct timeval tv;

	//Get the number of the user. number = Kbytes of the array.
	int size = atoi(argv[1]) * 1024 / 4;

	//warmup cache
	cacheWarmup(size, 10);
//	int* source_array;
	double time_start, time_taken;

	//naive copy
	int* source_array = new int[size];
	int iter = 0;
	for (int i = 0; i < size; ++i)
	{
		source_array[i] = i;
		iter = i;
	}

	printf("Source size len %u\nIterated: %u\nSource size bytes: %lu\n", source_array[size-1], iter, sizeof(source_array));

//	int* naive_array = new int[size];

//	gettimeofday(&tv, 0);
//	time_start = tv.tv_sec + 1e-6 * tv.tv_usec;
//	for(int i=0; i < size; ++i){
//		naive_array[i] = source_array[i];
//	}
//	memcpy(naive_array, source_array, sizeof(source_array));

//	gettimeofday(&tv, 0);
//	time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

//	printf("Time taken for naive copy: %f seconds.\n", time_taken);

	// //simd_memcpy
//	 for (int i = 0; i < size; ++i)
//	 {
//	 	source_array[i] = i;
//	 }
	 int* simd_array = new int[size];

	 gettimeofday(&tv, 0);
	 time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	 simd_memcpy(simd_array, source_array, sizeof(size*4));

	 gettimeofday(&tv, 0);
	 time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	 printf("Time taken for simd copy: %f seconds. item: %u\n", time_taken, simd_array[rand() % size]);

	//simd_memcpy_cache
	// for (int i = 0; i < size; ++i)
	// {
	// 	source_array[i] = i;
	// }
	// int simd_cache_array[size];

	// gettimeofday(&tv, 0);
	// time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	// simd_memcpy_cache(simd_cache_array, source_array, sizeof(source_array));

	// gettimeofday(&tv, 0);
	// time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	// printf("Time taken for simd cache copy: %f seconds.\n", time_taken);

  //checking for correctness of copy and mutability
  for(int i=0; i<size; ++i){
//    if(source_array[i] != naive_array[i]){
//	printf("Failed at: %u", i);
//      std::cout << "Failed naive_array copy.";
//      exit(EXIT_FAILURE);
//    }
     if(source_array[i] != simd_array[i]){
       std::cout << "Failed simd_array copy.";
       exit(EXIT_FAILURE);
     }
    // if(source_array[i] != simd_cache_array[i]){
    //   std::cout << "Failed simd_cache_array copy.";
    //   exit(EXIT_FAILURE);
    // }
  }

  source_array[0] = 42;

  //if(source_array[0] == naive_array[0]){
    //printf("%u to %u", source_array[0], naive_array[0]);
    //std::cout << "Failed naive_array mutability.";
    //exit(EXIT_FAILURE);
  //}
   if(source_array[0] == simd_array[0]){
     std::cout << "Failed simd_array mutability.";
     exit(EXIT_FAILURE);
   }
  // if(source_array[0] == simd_cache_array[0]){
  //   std::cout << "Failed simd_cache_array mutability.";
  //   exit(EXIT_FAILURE);
  // }
}

void cacheWarmup(int size, int iter)
{
	//create and populate array (32-bit, 4 byte slots)
	int*  arr = new int[size];

	for (int i = 0; i < size; ++i)
	{
		arr[i] = i;
	}
	
	//warm up cache
	int warmup_count = iter;

	for (int i = 0; i < warmup_count; ++i)
	{
		int* temp_arr = new int[size];
		memcpy(temp_arr, arr, sizeof(size*4));
	}
}


void simd_memcpy(void *dst, void *src, size_t nbytes)
{
  size_t i;

  size_t ilen = nbytes/sizeof(int);
  size_t ilen_sm = ilen - ilen%16;
  
  char *cdst=(char*)dst;
  char *csrc=(char*)src;

  int * idst=(int*) dst;
  int * isrc=(int*) src;

  __m128i l0,l1,l2,l3;

  _mm_prefetch((__m128i*)&isrc[0], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[4], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[8], _MM_HINT_NTA);
  _mm_prefetch((__m128i*)&isrc[12], _MM_HINT_NTA);
  
  for(i=0;i<ilen_sm;i+=16)
    {
      l0 =  _mm_load_si128((__m128i*)&isrc[i+0]);
      l1 =  _mm_load_si128((__m128i*)&isrc[i+4]);
      l2 =  _mm_load_si128((__m128i*)&isrc[i+8]);
      l3 =  _mm_load_si128((__m128i*)&isrc[i+12]);
    
      _mm_prefetch((__m128i*)&isrc[i+16], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+20], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+24], _MM_HINT_NTA);
      _mm_prefetch((__m128i*)&isrc[i+28], _MM_HINT_NTA);

      _mm_stream_si128((__m128i*)&idst[i+0],  l0);
      _mm_stream_si128((__m128i*)&idst[i+4],  l1);
      _mm_stream_si128((__m128i*)&idst[i+8],  l2);
      _mm_stream_si128((__m128i*)&idst[i+12], l3);

    }

  for(i=ilen_sm;i<ilen;i++)
    {
      idst[i] = isrc[i];
    }

  for(i=(4*ilen);i<nbytes;i++)
    {
      cdst[i] = csrc[i];
    }
}

void simd_memcpy_cache(void *dst, void *src, size_t nbytes)
{
  size_t i;
  size_t sm = nbytes - nbytes%sizeof(int);
  size_t ilen = nbytes/sizeof(int);
  size_t ilen_sm = ilen - ilen%16;

  //printf("nbytes=%zu,ilen=%zu,ilen_sm=%zu\n",
  //nbytes,ilen,ilen_sm);


  char *cdst=(char*)dst;
  char *csrc=(char*)src;

  int * idst=(int*)dst;
  int * isrc=(int*)src;

  __m128i l0,l1,l2,l3;

  _mm_prefetch((__m128i*)&isrc[0], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[4], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[8], _MM_HINT_T0);
  _mm_prefetch((__m128i*)&isrc[12], _MM_HINT_T0);
  
  for(i=0;i<ilen_sm;i+=16)
    {
      l0 =  _mm_load_si128((__m128i*)&isrc[i+0]);
      l1 =  _mm_load_si128((__m128i*)&isrc[i+4]);
      l2 =  _mm_load_si128((__m128i*)&isrc[i+8]);
      l3 =  _mm_load_si128((__m128i*)&isrc[i+12]);
    
      _mm_prefetch((__m128i*)&isrc[i+16], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+20], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+24], _MM_HINT_T0);
      _mm_prefetch((__m128i*)&isrc[i+28], _MM_HINT_T0);

      _mm_store_si128((__m128i*)&idst[i+0],  l0);
      _mm_store_si128((__m128i*)&idst[i+4],  l1);
      _mm_store_si128((__m128i*)&idst[i+8],  l2);
      _mm_store_si128((__m128i*)&idst[i+12], l3);

    }

  for(i=ilen_sm;i<ilen;i++)
    {
      idst[i] = isrc[i];
    }

  for(i=(ilen*4);i<nbytes;i++)
    {
      cdst[i] = csrc[i];
    }
}
