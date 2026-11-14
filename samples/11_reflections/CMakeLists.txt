add_executable(cyber_sample_reflections main.cpp)

target_link_libraries(cyber_sample_reflections PRIVATE
    cyber::software_renderer
    cyber::memory
)
target_compile_features(cyber_sample_reflections PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_sample_reflections)

if(CYBER_RUNTIME_OPTIONS)
    target_compile_options(cyber_sample_reflections PRIVATE ${CYBER_RUNTIME_OPTIONS})
endif()

set_target_properties(cyber_sample_reflections PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
