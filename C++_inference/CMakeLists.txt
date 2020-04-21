cmake_minimum_required(VERSION 3.5)

# Define project name
project(opencv_example_project)

SET(CMAKE_C_COMPILER g++)
add_definitions(--std=c++11)

# 指定libTorch位置
set(CMAKE_PREFIX_PATH  ./lib/libtorch)
find_package(Torch REQUIRED)

message(STATUS "Torch library status:")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBS}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "    torch lib : ${TORCH_LIBRARIES} ")

# 指定OpenCV位置
#set(OpenCV_DIR /run/media/geoffrey/Timbersaw/Backup/other_package/opencv-4.0.0/build)
# set(OpenCV_DIR /opt/opencv2)
find_package(OpenCV  REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    opencv lib : ${OpenCV_LIBRARIES} ")

# 包含头文件include
include_directories(${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# 生成的目标文件（可执行文件）
add_executable(main main.cpp)

# 置需要的库文件lib
# set(OpenCV_LIBS opencv_core  opencv_highgui opencv_imgcodecs opencv_imgproc)
target_link_libraries(main  ${OpenCV_LIBS} ${TORCH_LIBRARIES}) #