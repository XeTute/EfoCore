#include <iostream>
#include <chrono>
#include "ec.hpp"

float fun(const float&) { return 1.f; }

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

		con.resize(elems);
		con.setThreads(threads);

		con.adjustThreads();
		con.apply(fun, con);

		std::chrono::high_resolution_clock::time_point tp[2] = { std::chrono::high_resolution_clock::now() };
		con *= con;
		float sum = con.sum();
		tp[1] = std::chrono::high_resolution_clock::now();

		std::cout << "Sum: " << sum << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(tp[1] - tp[0]).count() << "ms";
		std::cout << "\n--- --- ---\n";
	}

	return 0;
}