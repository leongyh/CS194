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

	//seed rand gen with time
	srand(time(NULL));

	//Get the number of the user
	int N = atoi(argv[1]);

	int arr[N];

	for(int i = 0; i < N; i++)
	{
		arr[i] = i;
	}

	//randomize using fisher-yates shuffle
	for(int i = 0; i < N; i++)
	{
		int temp = arr[i]
		int rand_index = rand() % N + i

		arr[i] = arr[rand_index];
		arr[rand_index] = temp;
	}
	
	//timer for chase
	uint64_t current_time = getTicks(c1);

	//pointer chasing
	int j = 0;
	for(int i = 0; i < N; i++)
	{
		j = arr[j];
	}
	
	uint64_t elapsed_time = getTicks(c1) - current_time;

	printf("Time in Ticks to calculate sum of integers from 0 to 10000 (exclusive): %lu\n", elapsed_time);
}