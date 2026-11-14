add_executable(cyber_sample_clustered_lighting main.cpp)

target_link_libraries(cyber_sample_clustered_lighting PRIVATE
    cyber::core
    cyber::memory
    cyber::renderer
    cyber::rhi
    cyber::software_renderer
)
target_compile_features(cyber_sample_clustered_lighting PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_sample_clustered_lighting)

if(CYBER_RUNTIME_OPTIONS)
    target_compile_options(cyber_sample_clustered_lighting PRIVATE ${CYBER_RUNTIME_OPTIONS})
endif()

set_target_properties(cyber_sample_clustered_lighting PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
