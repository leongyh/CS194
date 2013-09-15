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
	int N = atoi(argv[1]) * 1024 / 4;

	int *arr;
	arr = (int*) calloc(N, sizeof(int));
	
	for(int i = 0; i < N; i++)
	{
		arr[i] = i;
	}

	int temp, rand_index;

	//randomize using fisher-yates shuffle
	for(int i = N-1; i > 0; i--)
	{
//		std::swap(arr[i], arr[rand()%(i+1));
		rand_index = rand() % (i+1);
		temp = arr[rand_index];

		arr[rand_index] = arr[i];
		arr[i] = temp;
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

	printf("Time in Ticks to chase an array of size %u: %lu\n", N, elapsed_time);
}
