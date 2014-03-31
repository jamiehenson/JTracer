JTracer
=======
by Jamie Henson

A recursive raytracer that uses cosine-weighted Monte Carlo hemispherical sampling.

Contains:
- Framework: Standard camera, multi-threading, external scene file loader
- Primitives: Plane, sphere, triangle
- Materials: Lambertian, specular, refractive
- Refractive materials: Glass, air, water, diamond
- Other: Ambient occlusion, global illumination, colour bleeding

Compile using:
g++ jtracer.cpp -O3 -std=c++11 -Wall -pedantic -fopenmp -o jtracer

Run using:
./jtracer SAMPLESIZE RESOLUTION SCENEFILE OUTPUTFILE*
(* denotes an optional parameter)

For example:
./jtracer 32 400 casino casinopic
Will export a 400x400 image at 32 samples per pixel from casino.scene, to casinopic.ppm

If you give no parameters, the program will use "default.scene" and output "default.ppm",
with a resolution of 512x512 at 16 samples per pixel.

Scene files are of type ".scene"

Based upon several small ray-tracer implementations.