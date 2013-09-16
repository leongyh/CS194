#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <time.h>
#include "counters.h"

void opt_simd_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar1_sgemm(float *Y, float *A, float *B, int n);
void opt_scalar0_sgemm(float *Y, float *A, float *B, int n);
void naive_sgemm(float *Y, float *A, float *B, int n);

int main(int argc, char *argv[])
{
	//init inst counter
	hwCounter_t c;
	c.init = false;
	initInsns(c);

	//cyccle counter
	hwCounter_t c1;
	c1.init = false;
	initTicks(c1);

	//time counter
	struct  timeval tv;
	double timestart, timetaken;
	
	int n = (1<<10);
	float* A = new float[n*n];
	float* B = new float[n*n];
	float* Y = new float[n*n];

	//seed rand gen with time
	srand(time(NULL));

	for(int i=0; i<n*n; ++i){
		A[i] = (float)rand()/(float)RAND_MAX;
		B[i] = (float)rand()/(float)RAND_MAX;
	}


	uint64_t current_time = getTicks(c1);
	uint64_t current_ins = getInsns(c);
	gettimeofday(&tv, 0);
	timestart = tv.tv_sec + 1e-6*tv.tv_usec;

	opt_scalar0_sgemm(Y, A, B, n);

	uint64_t exe_ins = getInsns(c) - current_ins;
	uint64_t elapsed_cyc = getTicks(c1) - current_time;
	gettimeofday(&tv, 0);
	timetaken = (tv.tv_sec + 1e-6*tv.tv_usec)-timestart;

	printf("Ins: %lu\n", exe_ins);
	printf("Cyc: %lu\n", elapsed_cyc);
	printf("Sec: %f\n", timetaken);
	printf("flops, ipc: %f, %lu\n", exe_ins/timetaken, exe_ins/elapsed_cyc);

	delete [] A;
	delete [] B;
	delete [] Y;
}
