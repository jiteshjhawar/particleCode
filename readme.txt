How to compile

open terminal. CD into src folder
Type
nvcc -lSDL2 -arch=sm_35 -rdc=true kernel.cu main.cpp Particle.cpp Swarm.cpp Store.cpp Screen.cpp -o alignTimeSeries -lcudadevrt

