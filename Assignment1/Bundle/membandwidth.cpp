#include <emmintrin.h>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <sys/time.h>
#include <time.h>
#include <cstdio>

void cacheWarmup(int size, int iter);
void simd_memcpy(void *dst, void *src, size_t nbytes);
void simd_memcpy_cache(void *dst, void *src, size_t nbytes);

int main(int argc, char *argv[])
{
	struct timeval tv;

	//Get the number of the user. number = Kbytes of the array.
	int size = atoi(argv[1]) * 1024 / 4;

	//warmup cache
	cacheWarmup(size, 1000);
	int* source_array;
	double time_start, time_taken;

	//naive copy
	source_array = (int*) malloc(size*sizeof(int));
	for (int i = 0; i < size; ++i)
	{
		source_array[i] = i;
	}
	int* naive_array = (int*) malloc(sizeof(source_array));

	gettimeofday(&tv, 0);
	time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	memcpy(naive_array, source_array, sizeof(source_array));

	gettimeofday(&tv, 0);
	time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for naive copy: %f seconds.\n", time_taken);

	//simd_memcpy
	for (int i = 0; i < size; ++i)
	{
		source_array[i] = i;
	}
	int* simd_array = (int*) malloc(sizeof(source_array));

	gettimeofday(&tv, 0);
	time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	simd_memcpy(simd_array, source_array, sizeof(source_array));

	gettimeofday(&tv, 0);
	time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for simd copy: %f seconds.\n", time_taken);

	//simd_memcpy_cache
	for (int i = 0; i < size; ++i)
	{
		source_array[i] = i;
	}
	int* simd_cache_array = (int*) malloc(sizeof(source_array));

	gettimeofday(&tv, 0);
	time_start = tv.tv_sec + 1e-6 * tv.tv_usec;

	memcpy(simd_cache_array, source_array, sizeof(source_array));

	gettimeofday(&tv, 0);
	time_taken = (tv.tv_sec + 1e-6 * tv.tv_usec) - time_start;

	printf("Time taken for simd cache copy: %f seconds.\n", time_taken);
}

void cacheWarmup(int size, int iter)
{
	//create and populate array (32-bit, 4 byte slots)
	int* arr = (int*) malloc(size * sizeof(int));

	for (int i = 0; i < size; ++i)
	{
		arr[i] = i;
	}
	
	//warm up cache
	int warmup_count = iter;

	for (int i = 0; i < warmup_count; ++i)
	{
		int* temp_arr = (int*) malloc(sizeof(arr));
		memcpy(temp_arr, arr, sizeof(arr));
	}
}

void simd_memcpy(void *dst, void *src, size_t nbytes)
{
  size_t i;

  size_t ilen = nbytes/sizeof(int);
  size_t ilen_sm = ilen - ilen%16;

  char *cdst=(char*)dst;
  char *csrc=(char*)src;

  int * idst=(int*)dst;
  int * isrc=(int*)src;

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