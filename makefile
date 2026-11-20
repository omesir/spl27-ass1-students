all: link

link: compile
	g++ -o bin/server bin/Main.o bin/Model.o bin/Server.o bin/GPU.o bin/MemoryBlock.o bin/Request.o bin/SchedulingPolicy.o

compile:
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Model.cpp -o bin/Model.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Main.cpp -o bin/Main.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Server.cpp -o bin/Server.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/GPU.cpp -o bin/GPU.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/MemoryBlock.cpp -o bin/MemoryBlock.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/Request.cpp -o bin/Request.o
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

clean:
	rm -rf bin/*