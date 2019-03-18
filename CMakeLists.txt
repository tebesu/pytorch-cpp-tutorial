cmake_minimum_required(VERSION 3.5)
project(pytorchtest)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
add_executable(train main.cpp)
#add_executable(train main.cpp FilenameDataset.h)

target_link_libraries(train "${TORCH_LIBRARIES}")

# For OpenCV dataloader
#find_package(OpenCV REQUIRED)
#target_link_libraries(train "${OpenCV_LIBS}")
