
# ClinkerForecast Core Makefile
CXX = g++
CXXFLAGS = -std=c++17 -O3 -ffast-math -funroll-loops -fomit-frame-pointer            -Wall -Wextra -Icore/include
LDFLAGS = -lm

# Источники
CORE_SRC = core/src/model.cpp            core/src/inference.cpp            core/src/export.cpp            core/src/physics.cpp            core/src/ops/ops.cpp

CLI_SRC = tools/cli/main.cpp
TEST_SRC = tests/test_load.cpp

# Объектные файлы
CORE_OBJ = $(CORE_SRC:.cpp=.o)
CLI_OBJ = $(CLI_SRC:.cpp=.o)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

# Цели
.PHONY: all clean test cli shared static

all: cli test

# Директории
$(shell mkdir -p build/lib build/bin build/tests)

# Компиляция ядра
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Shared library
shared: $(CORE_OBJ)
	$(CXX) -shared -o build/lib/libclinker_forecast.so $(CORE_OBJ) $(LDFLAGS)

# Static library
static: $(CORE_OBJ)
	ar rcs build/lib/libclinker_forecast.a $(CORE_OBJ)

# CLI
cli: static $(CLI_OBJ)
	$(CXX) -o build/bin/clinker_forecast $(CLI_OBJ) build/lib/libclinker_forecast.a $(LDFLAGS)

# Tests
test: static $(TEST_OBJ)
	$(CXX) -o build/tests/test_load $(TEST_OBJ) build/lib/libclinker_forecast.a $(LDFLAGS)
	./build/tests/test_load core/weights/model_v1.bin

# Benchmark
bench: cli
	./build/bin/clinker_forecast -m core/weights/model_v1.bin -b 1000

# Clean
clean:
	rm -f $(CORE_OBJ) $(CLI_OBJ) $(TEST_OBJ)
	rm -rf build/

# Install
install: all
	mkdir -p /usr/local/lib /usr/local/include /usr/local/bin
	cp build/lib/libclinker_forecast.so /usr/local/lib/
	cp core/include/clinker_forecast.h /usr/local/include/
	cp build/bin/clinker_forecast /usr/local/bin/
	ldconfig
