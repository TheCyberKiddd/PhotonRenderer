add_executable(cyber_sample_global_illumination main.cpp)

target_link_libraries(cyber_sample_global_illumination PRIVATE
    cyber::core
    cyber::jobs
    cyber::memory
    cyber::renderer
    cyber::rhi
    cyber::software_renderer
)
target_compile_features(cyber_sample_global_illumination PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_sample_global_illumination)

if(CYBER_RUNTIME_OPTIONS)
    target_compile_options(cyber_sample_global_illumination PRIVATE ${CYBER_RUNTIME_OPTIONS})
endif()

set_target_properties(cyber_sample_global_illumination PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
