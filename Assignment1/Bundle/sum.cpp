#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include "counters.h"

int main(int argc, char *argv[])
{
	hwCounter_t c1;
	c1.init = false;
	initTicks(c1);

	//timer for sum
	uint64_t current_time = getTicks(c1);

	//Get the number of the user
	int n = atoi(argv[1]);

	//Compute sum
	long long sum = 0;
	for (long long i = 0; i < n; i++)
	{
		sum += 1;
	}

	printf("Total sum is: %llu\n", sum);
	
	uint64_t elapsed_time = getTicks(c1) - current_time;

	printf("Time in Ticks to calculate sum of integers from 0 to 10000 (exclusive): %lu\n", elapsed_time);
}