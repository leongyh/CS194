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