#include <iostream>
#include <chrono>
#include <omp.h>
#include "ec.hpp"


int main()
{
	EC::ec con(100 * 1000000); // x * 1M
	EC::n threads = 6;
	con.compile(threads);

	omp_set_num_threads(6);

#pragma omp parallel for
	for (std::intmax_t i = 0; i < signed(con.size()); ++i)
		con[i] = 15.f;

	std::chrono::high_resolution_clock::time_point tp[4];
	long long td[2];

	tp[0] = {std::chrono::high_resolution_clock::now()};
	for (EC::n i = 0; i < con.size(); ++i)
		con[i] *= con[i];
	tp[1] = std::chrono::high_resolution_clock::now();
	td[0] = std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count();
	std::cout << "Naive took " << td[0] << "ms.\n";

	for (EC::n i = 0; i < con.size(); ++i)
		con[i] = 15.f;

	tp[0] = std::chrono::high_resolution_clock::now();
	con.square(threads);
	tp[1] = std::chrono::high_resolution_clock::now();
	td[1] = std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count();
	std::cout << "EC took " << td[1] << "ms.\n";

	std::cout << "EC was faster by a factor of " << float(td[0]) / float(td[1]) << ".\n";
	std::cout << "Last Element: " << con[con.size() - 1] << '\n';

	return 0;
}