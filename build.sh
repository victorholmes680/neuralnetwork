#!/bin/sh
set -xe
CFLAGS="-O3 -Wall -Wextra `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm"

cc $CFLAGS -o adder_gen adder_gen.c $LIBS
cc $CFLAGS -o gym gym.c $LIBS
cc $CFLAGS -o img2nn img2nn.c $LIBS

