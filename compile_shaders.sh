#!/bin/sh

shopt -s nullglob

rm -r shaders/compiled/*.spv

for file in shaders/*.{vert,frag}
do
glslc $file -o shaders/compiled/$(basename $file).spv
done

for file in shaders/compiled/*.spv
do
spirv-opt $file -O -o $file
done
