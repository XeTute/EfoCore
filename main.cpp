#include <iostream>
#include <chrono>
#include <omp.h>
#include "ec.hpp"


int main()
{
	while (true)
	{
		EC::ec con;

		EC::n nthreads;
		EC::n nelems;

		std::cout << "nthreads: ";
		std::cin >> nthreads;

		std::cout << "nelems ( / 1 000 000): ";
		std::cin >> nelems;

		nelems *= 1000000;
		con.resize(nelems);

		omp_set_num_threads(nthreads);

		for (EC::n i = 0; i < signed(con.size()); ++i)
			con[i] = 15.25f;

		std::chrono::high_resolution_clock::time_point tp[4];
		long long td[2];

		tp[0] = { std::chrono::high_resolution_clock::now() };
#pragma omp parallel for
		for (std::intmax_t i = 0; i < con.size(); ++i)
			con[i] *= con[i];
		tp[1] = std::chrono::high_resolution_clock::now();
		td[0] = std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count();
		std::cout << "OpenMP took " << td[0] << "ms.\n";

		for (EC::n i = 0; i < con.size(); ++i)
			con[i] = 15.25f;

		tp[0] = std::chrono::high_resolution_clock::now();
		con.mult(con);
		tp[1] = std::chrono::high_resolution_clock::now();
		td[1] = std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count();
		std::cout << "EC took     " << td[1] << "ms.\n";

		if (td[0] < td[1])
			std::cout << "OpenMP was faster than EC.\n";
		else if (td[0] == td[1])
			std::cout << "OpenMP was as fast as EC.\n";
		else
			std::cout << "EC was faster than OpenMP.\n";
		std::cout << "--- --- ---" << std::endl;
	}

	return 0;
}