cmake_minimum_required(VERSION 3.24)
project(Neural_Network_IP)

set(CMAKE_CXX_STANDARD 17)

add_executable(Neural_Network_IP main.cpp neural_network.cpp mnist_loader.cpp)
