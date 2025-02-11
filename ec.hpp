#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <algorithm>
#include <omp.h>

#ifndef EC_HPP
#define EC_HPP

namespace EC
{
	using n = std::size_t;
	using _n = std::intmax_t;

	struct mul
	{
		__m256 operator()(__m256 a, __m256 b) const { return _mm256_mul_ps(a, b); }
		float operator()(float a, float b) const { return a * b; }
	};

	struct div
	{
		__m256 operator()(__m256 a, __m256 b) const { return _mm256_div_ps(a, b); }
		float operator()(float a, float b) const { return a + b; }
	};

	struct add
	{
		__m256 operator()(__m256 a, __m256 b) const { return _mm256_add_ps(a, b); }
		float operator()(float a, float b) const { return a + b; }
	};

	struct sub
	{
		__m256 operator()(__m256 a, __m256 b) const { return _mm256_sub_ps(a, b); }
		float operator()(float a, float b) const { return a + b; }
	};

	// template <__m256 (*SIMDfun) (__m256, __m256)>: Inst. don't have pointers.
	template <typename opr>
	void doSIMD(n offsetchunk, const n& chunksize, float* to, float* by, const opr& fun)
	{
		_n i = offsetchunk * chunksize;
		_n im = i + (chunksize / 8) * 8;
		_n remm = i + chunksize;

		for (; i < im; i += 8)
		{
			float* tmpptr = &to[i];
			__m256 vec0 = _mm256_load_ps(tmpptr);
			__m256 vec1 = _mm256_load_ps(&by[i]);

			vec0 = fun(vec0, vec1);
			_mm256_storeu_ps(tmpptr, vec0);
		}

		for (; i < remm; ++i)
			to[i] = fun(to[i], by[i]);
	}

	class ec
	{
	private:
		float* mem;

		n elems;

		mul _mul;
		div _div;
		add _add;
		sub _sub;

	public:
		n threads;

		ec() : mem(nullptr), elems(0), threads(1) {}
		ec(n _elems) { resize(_elems); }

		void resize(n _elems)
		{
			elems = _elems;
			if (mem) _aligned_free(mem);
			mem = (float*)_aligned_malloc(elems * sizeof(float), 32);
			threads = 1;
		}

		const n& size() { return elems; }
		const float* data() { return mem; }

		const EC::ec& operator* (const EC::ec& other)
		{
			EC::ec tmp(elems);
			std::copy(this->mem, this->mem + elems, tmp.mem);

			doSIMD(0, elems, tmp.mem, other.mem, _mul);
			return tmp;
		}

		const EC::ec& operator/ (const EC::ec& other)
		{
			EC::ec tmp(elems);
			std::copy(this->mem, this->mem + elems, tmp.mem);
		
			doSIMD(0, elems, tmp.mem, other.mem, _div);
			return tmp;
		}
		
		const EC::ec& operator+ (const EC::ec& other)
		{
			EC::ec tmp(elems);
			std::copy(this->mem, this->mem + elems, tmp.mem);
		
			doSIMD(0, elems, tmp.mem, other.mem, _add);
			return tmp;
		}
		
		const EC::ec& operator- (const EC::ec& other)
		{
			EC::ec tmp(elems);
			std::copy(this->mem, this->mem + elems, tmp.mem);
		
			doSIMD(0, elems, tmp.mem, other.mem, _sub);
			return tmp;
		}

		void operator*= (const EC::ec& other) { doSIMD(0, elems, this->mem, other.mem, _mul); }
		void operator/= (const EC::ec& other) { doSIMD(0, elems, this->mem, other.mem, _div); }
		void operator+= (const EC::ec& other) { doSIMD(0, elems, this->mem, other.mem, _add); }
		void operator-= (const EC::ec& other) { doSIMD(0, elems, this->mem, other.mem, _sub); }
		float& operator[] (n i) { return mem[i]; }

		~ec()
		{
			if (mem) _aligned_free(mem);
		}
	};
};
#endif
