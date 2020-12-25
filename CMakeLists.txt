cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(main)

find_package(Torch REQUIRED)
find_package(Boost REQUIRED system)
find_package(Boost COMPONENTS filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Find all executables
file(GLOB project_SRCS src/*.cpp) #src/*.h

add_executable(main ${project_SRCS})
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main ${Boost_SYSTEM_LIBRARY})
target_link_libraries(main ${Boost_FILESYSTEM_LIBRARY})
set_property(TARGET main PROPERTY CXX_STANDARD 17)