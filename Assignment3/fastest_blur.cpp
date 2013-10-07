#include "pmmintrin.h"
#include <omp.h>
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

void fastest_blur_block(float* out, int rs, int cs, int re, int ce, int n, float* frame, int* radii){
	__m128 v1,v2,v3,v4;
	float* temp = new float[4];

	for(int r=rs; r<re; r++){
		for(int c=cs; c<ce; c++){
			int rd = radii[r*n+c];
			float avg = 0;

			int row_start = max(0,r-rd);
			int col_start = max(0,c-rd);
			int row_size = min(n-1, r+rd) - row_start + 1;
			int col_size = min(n-1, c+rd) - col_start + 1;
			int size = row_size * col_size;

			for(int row_batch = 0; row_batch < row_size/4; row_batch++){
				for(int col_batch = 0; col_batch < col_size/4; col_batch++){
					v1 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 0)*n + (col_start + col_batch*4)]);
					v2 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 1)*n + (col_start + col_batch*4)]);
					v3 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 2)*n + (col_start + col_batch*4)]);
					v4 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 3)*n + (col_start + col_batch*4)]);

					__m128 sum1 = _mm_hadd_ps(v1, v2);
					__m128 sum2 = _mm_hadd_ps(v3, v4);
					__m128 sum = _mm_hadd_ps(sum1, sum2);

					_mm_store_ps((float*)&temp[0], sum);

					avg += (temp[0] + temp[1] + temp[2] + temp[3]);
				}
			}

			int row_batch = row_size/4;
			
			switch(row_size % 4)
			{
				case 3:
					for(int col_batch = 0; col_batch < col_size/4; col_batch++){
						v1 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 0)*n + (col_start + col_batch*4)]);
						v2 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 1)*n + (col_start + col_batch*4)]);
						v3 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 2)*n + (col_start + col_batch*4)]);

						__m128 sum1 = _mm_hadd_ps(v1, v2);
						__m128 sum = _mm_hadd_ps(sum1, v3);

						_mm_store_ps((float*)&temp[0], sum);

						avg += temp[0] + temp[1] + temp[2] + temp[3];
					}
					break;

				case 2:
					for(int col_batch = 0; col_batch < col_size/4; col_batch++){
						v1 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 0)*n + (col_start + col_batch*4)]);
						v2 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 1)*n + (col_start + col_batch*4)]);
							
						__m128 sum = _mm_hadd_ps(v1, v2);

						_mm_store_ps((float*)&temp[0], sum);

						avg += (temp[0] + temp[1] + temp[2] + temp[3]);
					}
					break;

				case 1:
					for(int col_batch = 0; col_batch < col_size/4; col_batch++){
						v1 = _mm_loadu_ps((float*)&frame[(row_start + row_batch*4 + 0)*n + (col_start + col_batch*4)]);
					 
						__m128 sum = _mm_hadd_ps(v1, v1);

						_mm_store_ps((float*)&temp[0], sum);

						avg += (temp[0] + temp[1]);

					}
					break;

				case 0:
					avg += 0;
			}

			for(int row_single = row_start; row_single < row_start + row_size;  row_single++){
				for(int col_single = col_start + col_size/4 * 4; col_single < col_start + col_size; col_single++){
					avg += frame[row_single*n + col_single];
				}
			}

			out[r*n+c] = avg/size;
		}
	}
}

void fastest_blur(float* out, int n, float* frame, int* radii, int nthr){
	omp_set_num_threads(nthr);

	int num_tasks = 1000;
	int load_size = (n*n)/num_tasks;

		#pragma omp parallel
		{
			for(int r=0; r*load_size<(n/load_size)*load_size; r++)
				for(int c=0; c*load_size<(n/load_size)*load_size; c++)
				{
					#pragma omp task
						fastest_blur_block(out, r*load_size, c*load_size, (r+1)*load_size, (c+1)*load_size, n, frame, radii);
				}
		}

		for(int r=0; r<n; r++)
			for(int c=(n/load_size)*load_size; c<n; c++)
			{
				fastest_blur_block(out, r, c, n, frame, radii);
			}

		for(int r=(n/load_size)*load_size; r<n; r++)
			for(int c=0; c<(n/load_size)*load_size; c++)
			{
				fastest_blur_block(out, r, c, n, frame, radii);
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

	printf("Time needed for naive blur = %.3f seconds.\n", time);

	//Blur using vector blur
	for(int nthr = 1; nthr <= 16; nthr++){
		float* out2 = new float[n*n];
		double time2 = timestamp();
		fastest_blur(out2, n, frame, radii, nthr);
		time2 = timestamp() - time2;

		//Check result
		for(int i=0; i<n; i++)
			for(int j=0; j<n; j++){
				float dif = out[i*n+j] - out2[i*n+j];
				if(dif*dif>1.0f){
					printf("Your blur does not give the right result!\n");
					printf("For element (row, column, radii) = (%d, %d, %d):\n", i, j, radii[i*n+j]);
					printf("  Simple blur gives %.2f\n", out[i*n+j]);
					printf("  Your blur gives %.2f\n", out2[i*n+j]);
					exit(-1);
				}
		}
		
		delete[] out2;

		//Print out Time
		printf("Time needed for parallel blur with %d threads = %.3f seconds.\n", nthr, time2);
	}

	//Delete
	delete[] radii;
	delete[] frame;
	delete[] out;
}
