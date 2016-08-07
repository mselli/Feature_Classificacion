#C compiler
# CC = /usr/local/bin/clang-omp++ -O3 -std=c++11
CC = g++ -O3 -std=c++11

# SRC = main.cpp 
SRC = ./*.cpp


# CFLAGS = `pkg-config --cflags opencv` -fopenmp
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv` -lboost_system -lboost_filesystem -Ilibsvm -O3 libsvm/svm.cpp

EXE = main

release:$(SRC)
	$(CC)    $(SRC) $(LIBS) $(CFLAGS) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE_X) $(EXE).linkinfo 