.PHONY: all compile

all: compile

compile:
	nvcc -o proj proj.cu `pkg-config --cflags --libs opencv4`
