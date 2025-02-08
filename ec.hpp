#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <type_traits>
#include <thread>
#include <valarray>
#include <vector>

#ifndef EC_HPP
#define EC_HPP

namespace EC
{
	using n = std::size_t;

	void mult(n offsetchunk, const n& chunksize, float* multto, float* multby)
	{
		n start = offsetchunk * chunksize;
		n im = start + chunksize;

		for (n i = start; i < im; i += 8)
		{
			n iahead = i + 16;
			_mm_prefetch((char*)&multto[iahead], _MM_HINT_T0);
			_mm_prefetch((char*)&multby[iahead], _MM_HINT_T0);


			float* tmpptr = &multto[i];
			__m256 vec0 = _mm256_load_ps(tmpptr);
			__m256 vec1 = _mm256_load_ps(&multby[i]);

			vec0 = _mm256_mul_ps(vec0, vec1);
			_mm256_storeu_ps(tmpptr, vec0);
		}
	}

	class ec
	{
	private:
		
		float* mem;

		n elems;
		n chunksize;

	public:
		
		ec() : mem(nullptr), elems(0), chunksize(0) {};
		ec(n _elems) { resize(_elems); }

		void resize(n _elems)
		{
			elems = _elems;
			if (mem) _aligned_free(mem);
			mem = (float*)_aligned_malloc(elems * sizeof(float), sizeof(float));
		}

		void compile(const n& threads)
		{
			chunksize = elems / threads;
		}

		void square(const n& threads)
		{
			std::vector<std::thread> pool(threads);
			n dt = threads - 1;
			for (n t = 0; t < dt; ++t)
				pool[t] = std::thread(mult, t, chunksize, mem, mem);
			pool[dt] = std::thread(mult, dt, (chunksize * threads) == elems ? chunksize : (chunksize + 1), mem, mem);
			for (std::thread& t : pool)
				if (t.joinable()) t.join();
		}

		const n& size() { return elems; }
		const float* data() { return mem; }
		float& operator[] (n i) { return mem[i]; }

		~ec() { if (mem) _aligned_free(mem); }
	};
};
#endif