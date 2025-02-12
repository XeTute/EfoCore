#include <iostream>
#include <chrono>
#include <omp.h>
#include "ec.hpp"


int main()
{
	while (true)
	{
		EC::ec con;
		EC::n threads, elems;

		std::cout << "Threads: ";
		std::cin >> threads;
		std::cout << "Elems: ";
		std::cin >> elems;

		elems *= 1024;
		con.resize(elems);
		con.setThreads(threads);

#pragma omp parallel for
		for (EC::_n i = 0; i < elems; ++i)
			con[i] = 0.5f;

		std::cout << "Sum: " << con.sum();;
		std::cout << "\n--- --- ---\n";
	}

	return 0;
}