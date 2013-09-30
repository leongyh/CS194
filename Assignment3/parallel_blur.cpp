#include <emmintrin.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

using namespace std;

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

// Simple Blur
void simple_blur(float* out, int n, float* frame, int* radii){
  for(int r=0; r<n; r++)
    for(int c=0; c<n; c++){
      int rd = radii[r*n+c];
      int num = 0;
      float avg = 0;
      for(int r2=max(0,r-rd); r2<=min(n-1, r+rd); r2++)
        for(int c2=max(0, c-rd); c2<=min(n-1, c+rd); c2++){
          avg += frame[r2*n+c2];
          num++;
        }
      out[r*n+c] = avg/num;
    }
}

void parallel_blur(float* out, int n, float* frame, int* radii){
	__m128i v1,v2,v3,v4;

	for(int r=0; r<n; r++){
		for(int c=0; c<n; c++){
			int rd = radii[r*n+c];
			float avg = 0;

			int row_start = min(n-1, r+rd);
			int col_start = min(n-1, c+rd);
			int row_size = row_start - max(0,r-rd) + 1;
			int col_size = col_start - max(0,c-rd) + 1;
			int size = row_size * col_size

			for(int row_batch = 0; row_batch < row_size/4; row_batch++){
				for(int col_batch = 0; col_batch < col_size/4; col_batch++){
					uint32_t temp[4];

					v1 = _mm_load_si128((__m128i*)&frame[(row_start + row_batch*4 + 1)*n + (col_start + col_batch*4)]);
					v2 = _mm_load_si128((__m128i*)&frame[(row_start + row_batch*4 + 2)*n + (col_start + col_batch*4)]);
					v3 = _mm_load_si128((__m128i*)&frame[(row_start + row_batch*4 + 3)*n + (col_start + col_batch*4)]);
					v4 = _mm_load_si128((__m128i*)&frame[(row_start + row_batch*4 + 4)*n + (col_start + col_batch*4)]);

					__m128i sum1 = _mm_hadd_ps(v1, v2);
					__m128i sum2 = _mm_hadd_ps(v3, v4);
					__m128i sum = _mm_hadd_ps(sum1, sum2);
					sum = _mm_hadd_ps(sum, sum);

					_mm_store_si128((__m128i*)temp, sum);

					avg += temp[0] + temp[1];

					delete [] temp;
				}
			}

			for(int row_single = row_size/4 * 4; i < row_start + row_size; ++row_single){
				for (int col_single = col_start/4 * 4; i < col_start + col_size; ++col_single){
					avg += frame[row_single*n + col_single]; 
				}
			}

			out[r*n+c] = avg/size;
		}
	}
}

int main(int argc, char *argv[])
{
  //Generate random radii
  srand(0);
  int n = 3000;
  int* radii = new int[n*n];
  for(int i=0; i<n*n; i++)
    radii[i] = 6*i/(n*n) + rand()%6;

  //Generate random frame
  float* frame = new float[n*n];
  for(int i=0; i<n*n; i++)
    frame[i] = rand()%256;

  //Blur using simple blur
  float* out = new float[n*n];
  double time = timestamp();
  simple_blur(out, n, frame, radii);
  time = timestamp() - time;

  //Blur using vector blur
  float* out2 = new float[n*n];
  double time2 = timestamp();
  parallel_blur(out2, n, frame, radii);
  time2 = timestamp() - time2;

  //Check result
  for(int i=0; i<n; i++)
    for(int j=0; j<n; j++){
      float dif = out[i*n+j] - out2[i*n+j];
      if(dif*dif>1.0f){
        printf("Your blur does not give the right result!\n");
        printf("For element (row, column) = (%d, %d):\n", i, j);
        printf("  Simple blur gives %.2f\n", out[i*n+j]);
        printf("  Your blur gives %.2f\n", out2[i*n+j]);
        exit(-1);
      }
  }

  //Delete
  delete[] radii;
  delete[] frame;
  delete[] out;
  delete[] out2;

  //Print out Time
  printf("Time needed for naive blur = %.3f seconds.\n", time);
  printf("Time needed for your blur = %.3f seconds.\n", time2);
}
