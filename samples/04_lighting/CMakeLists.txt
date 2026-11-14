add_executable(cyber_sample_lighting main.cpp reference_renderer.cpp)

target_link_libraries(cyber_sample_lighting PRIVATE
    cyber::software_renderer
    cyber::platform
    cyber::memory
    cyber::jobs
)
target_compile_features(cyber_sample_lighting PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_sample_lighting)

if(CYBER_RUNTIME_OPTIONS)
    target_compile_options(cyber_sample_lighting PRIVATE ${CYBER_RUNTIME_OPTIONS})
endif()

set_target_properties(cyber_sample_lighting PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

cyber_add_test(
    NAME lighting_reference
    SOURCES tests/path_tracer_test.cpp reference_renderer.cpp
    DEPENDS software_renderer memory
)
