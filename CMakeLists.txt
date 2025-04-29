cmake_minimum_required(VERSION 3.10)
project(MezeraNeuron)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Включаем папку с заголовками
include_directories(include)

# Добавляем исходники
add_library(mezera_core
    src/core/Neuron.cpp
    src/core/Perceptron.cpp
)

# Пример main.cpp
add_executable(mezera_demo demo/main.cpp)

# Линкуем библиотеку к демонстрации
target_link_libraries(mezera_demo mezera_core)
