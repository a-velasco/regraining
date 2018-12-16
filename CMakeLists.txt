cmake_minimum_required(VERSION 3.10)
project(Regraining)
find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC src)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
