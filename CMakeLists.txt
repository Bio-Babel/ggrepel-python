cmake_minimum_required(VERSION 3.18)
project(ggrepel_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_repel MODULE
    ggrepel_py/_repel/repel_boxes.cpp
)

target_compile_options(_repel PRIVATE
    -O3 -Wall -Wextra -Wno-unused-parameter
)

install(TARGETS _repel
    LIBRARY DESTINATION ggrepel_py/_repel
    RUNTIME DESTINATION ggrepel_py/_repel
)
