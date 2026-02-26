#!/bin/sh
set -xe
cc -o xor xor.c -lm && ./xor
#cc -o adder adder.c -lm && ./adder

