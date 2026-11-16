cyber_add_module(
    NAME renderer
    LAYER 3
    SOURCES
        src/bindless_table.cpp
        src/clustered_lighting.cpp
        src/descriptor_allocator.cpp
        src/frame_stream.cpp
        src/gpu_texture_pool.cpp
        src/gpu_virtual_texture.cpp
        src/gpu_virtual_texture_feedback.cpp
        src/gpu_culling_pipeline.cpp
        src/gpu_driven.cpp
        src/gpu_light_culling.cpp
        src/hot_pipeline.cpp
        src/material.cpp
        src/material_file.cpp
        src/meshlet.cpp
        src/raytracing_path.cpp
        src/reflected_layout.cpp
        src/render_graph.cpp
        src/render_thread.cpp
        src/shadow_pass.cpp
        src/texture_streaming.cpp
        src/virtual_texture.cpp
    PUBLIC_HEADERS
        include/cyber/renderer/bindless_table.h
        include/cyber/renderer/clustered_lighting.h
        include/cyber/renderer/descriptor_allocator.h
        include/cyber/renderer/frame_stream.h
        include/cyber/renderer/gpu_texture_pool.h
        include/cyber/renderer/gpu_virtual_texture.h
        include/cyber/renderer/gpu_virtual_texture_feedback.h
        include/cyber/renderer/gpu_culling_pipeline.h
        include/cyber/renderer/gpu_driven.h
        include/cyber/renderer/gpu_light_culling.h
        include/cyber/renderer/hot_pipeline.h
        include/cyber/renderer/material.h
        include/cyber/renderer/meshlet.h
        include/cyber/renderer/raytracing_path.h
        include/cyber/renderer/reflected_layout.h
        include/cyber/renderer/render_graph.h
        include/cyber/renderer/render_thread.h
        include/cyber/renderer/shadow_pass.h
        include/cyber/renderer/texture_streaming.h
        include/cyber/renderer/virtual_texture.h
    DEPENDS core math memory jobs platform perf rhi shaders
)

cyber_add_test(
    NAME renderer_reflected_layout
    SOURCES
        tests/reflected_layout_test.cpp
    DEPENDS renderer shaders rhi memory
)

cyber_add_test(
    NAME renderer_hot_pipeline
    SOURCES
        tests/hot_pipeline_test.cpp
    DEPENDS renderer shaders rhi jobs memory
)

cyber_add_test(
    NAME renderer_public_headers
    SOURCES
        tests/public_headers_test.cpp
    DEPENDS renderer
)

cyber_add_test(
    NAME renderer_clustered_lighting
    SOURCES
        tests/clustered_lighting_test.cpp
    DEPENDS renderer memory
)

cyber_add_test(
    NAME renderer_material
    SOURCES
        tests/material_test.cpp
    DEPENDS renderer shaders rhi memory
)

cyber_add_test(
    NAME renderer_material_file
    SOURCES
        tests/material_file_test.cpp
    DEPENDS renderer
)

cyber_add_test(
    NAME renderer_phase19
    SOURCES
        tests/phase19_test.cpp
    DEPENDS renderer rhi memory
)

cyber_add_test(
    NAME renderer_render_graph
    SOURCES
        tests/render_graph_test.cpp
    DEPENDS renderer rhi memory
)

cyber_add_test(
    NAME renderer_shadow_pass
    SOURCES
        tests/shadow_pass_test.cpp
    DEPENDS renderer rhi memory
)

cyber_add_test(
    NAME renderer_render_thread
    SOURCES
        tests/render_thread_test.cpp
    DEPENDS renderer rhi memory platform
)

add_executable(cyber_renderer_clustered_lighting_bench benchmarks/clustered_lighting_bench.cpp)
target_link_libraries(cyber_renderer_clustered_lighting_bench PRIVATE cyber_renderer cyber_memory)

cyber_add_test(
    NAME renderer_raytracing_path
    SOURCES
        tests/raytracing_path_test.cpp
    DEPENDS renderer rhi memory
)

cyber_add_test(
    NAME renderer_gpu_driven
    SOURCES
        tests/gpu_driven_test.cpp
    DEPENDS renderer scene jobs math memory rhi
)

cyber_add_test(
    NAME renderer_gpu_driven_d3d12
    SOURCES
        tests/gpu_driven_d3d12_test.cpp
    DEPENDS renderer scene jobs math memory rhi
)

cyber_add_test(
    NAME renderer_texture_streaming
    SOURCES
        tests/texture_streaming_test.cpp
    DEPENDS renderer platform memory
)

cyber_add_test(
    NAME renderer_gpu_texture_pool
    SOURCES
        tests/gpu_texture_pool_test.cpp
    DEPENDS renderer rhi platform memory
)

cyber_add_test(
    NAME renderer_virtual_texture
    SOURCES
        tests/virtual_texture_test.cpp
    DEPENDS renderer platform memory
)

cyber_add_test(
    NAME renderer_gpu_virtual_texture
    SOURCES
        tests/gpu_virtual_texture_test.cpp
    DEPENDS renderer rhi platform memory
)

cyber_add_test(
    NAME renderer_bc6h_hardware
    SOURCES
        tests/bc6h_hardware_test.cpp
    DEPENDS renderer rhi assets platform memory
)

cyber_add_test(
    NAME renderer_gpu_virtual_texture_feedback
    SOURCES
        tests/gpu_virtual_texture_feedback_test.cpp
    DEPENDS renderer rhi platform memory
)

cyber_add_test(
    NAME renderer_gpu_streaming_render
    SOURCES
        tests/gpu_streaming_render_test.cpp
    DEPENDS renderer rhi platform memory
)

if(CYBER_BUILD_TESTS)
    set_tests_properties(renderer_gpu_streaming_render PROPERTIES TIMEOUT 180)
    set_tests_properties(renderer_gpu_virtual_texture PROPERTIES TIMEOUT 600)
    set_tests_properties(renderer_gpu_virtual_texture_feedback PROPERTIES TIMEOUT 600)
    set_tests_properties(renderer_bc6h_hardware PROPERTIES TIMEOUT 300)
    set_tests_properties(renderer_texture_streaming PROPERTIES TIMEOUT 60)
    set_tests_properties(renderer_gpu_texture_pool PROPERTIES TIMEOUT 120)
    set_tests_properties(renderer_virtual_texture PROPERTIES TIMEOUT 120)
    # The GPU-timing test measures CPU cost against object count; other GPU work
    # running beside it on a four-core machine distorts the ratio it checks.
    set_tests_properties(renderer_gpu_driven_d3d12 PROPERTIES RUN_SERIAL TRUE TIMEOUT 600)
    # The streaming tests rely on IO threads and a wall-clock frame-time bound
    # (p99); other GPU tests beside them on a four-core machine starve both.
    set_tests_properties(renderer_gpu_streaming_render PROPERTIES RUN_SERIAL TRUE)
    set_tests_properties(renderer_gpu_texture_pool PROPERTIES RUN_SERIAL TRUE)
endif()

# Phase 34: every GPU test again on Vulkan, same binary, same assertions.
foreach(gpu_test IN ITEMS
        renderer_phase19
        renderer_shadow_pass
        renderer_raytracing_path
        renderer_gpu_driven_d3d12
        renderer_gpu_texture_pool
        renderer_gpu_virtual_texture
        renderer_bc6h_hardware
        renderer_gpu_virtual_texture_feedback
        renderer_gpu_streaming_render)
    cyber_add_vulkan_variant(${gpu_test})
endforeach()
