cmake_minimum_required(VERSION 3.16)
project(Neuron_Networking)

set(CMAKE_CXX_STANDARD 14)

add_executable(Neuron_Networking main.cpp Neuron.cpp Neuron.h NeuronNetwork.cpp NeuronNetwork.h)