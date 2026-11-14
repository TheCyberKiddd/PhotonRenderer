cmake_minimum_required(VERSION 3.28)

project(CyberGraphicsPackageConsumer LANGUAGES C CXX)

find_package(CyberGraphics 0.1 CONFIG REQUIRED)

add_executable(consumer main.cpp c_api_consumer.c)
target_link_libraries(consumer PRIVATE
    cyber::core
    cyber::math
    cyber::memory
    cyber::platform
    cyber::perf
    cyber::jobs
    cyber::scene
    cyber::rhi
    cyber::renderer
    cyber::software_renderer
    cyber::graphics_api
    cyber::plugin_host
)
target_compile_features(consumer PRIVATE cxx_std_20)
