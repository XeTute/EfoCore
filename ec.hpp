#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <omp.h>

#ifndef EC_HPP
#define EC_HPP

namespace EC
{
	using n = std::size_t;
	using _n = std::intmax_t;

	void mult(n offsetchunk, const n& chunksize, float* multto, float* multby)
	{
		_n start = offsetchunk * chunksize;
		_n im = start + chunksize;

#pragma omp parallel for
		for (_n i = start; i < im; i += 8)
		{
			n iahead = i + 8;
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

	public:
		ec() : mem(nullptr), elems(0) {}
		ec(n _elems) { resize(_elems); }

		void resize(n _elems)
		{
			elems = _elems;
			if (mem) _aligned_free(mem);
			mem = (float*)_aligned_malloc(elems * sizeof(float), 32);
		}

		void mult(const EC::ec& with)
		{
			EC::mult(0, elems, mem, with.mem);
		}

		const n& size() { return elems; }
		const float* data() { return mem; }
		float& operator[] (n i) { return mem[i]; }

		~ec()
		{
			if (mem) _aligned_free(mem);
		}
	};
};
#endif