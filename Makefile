CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
AR ?= ar

SRC := src/status.cc src/core_algos_cpu.cc
OBJ := $(SRC:.cc=.o)
LIB := libcverl.a
TEST_BIN := build/test_core_algos_cpu
GOLDEN_BIN := build/compare_golden

.PHONY: all test clean

all: $(LIB) $(TEST_BIN) $(GOLDEN_BIN)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(TEST_BIN): tests/test_core_algos_cpu.cc $(LIB)
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

$(GOLDEN_BIN): tools/compare_golden.cc $(LIB)
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB) -o $@

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(OBJ) $(LIB)
	rm -rf build
