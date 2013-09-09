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
	long sum = sumRange(0, 10000);
	uint64_t elapsed_time = getTicks(c1) - current_time;

	printf("Time in Ticks to calculate sum of integers from 0 to 10000 (exclusive): %lu\n", (elapsed_time);
}

int sumRange(int from, int to_excl)
{
	long sum = 0;

	for(int i = from; i < to_excl; i++)
	{
		sum += i;
	}

	return sum
}