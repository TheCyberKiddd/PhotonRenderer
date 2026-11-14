set(CYBER_SOFTWARE_SOURCES
    src/antialias.cpp
    src/camera.cpp
    src/clustered_pbr.cpp
    src/debug_text.cpp
    src/denoise.cpp
    src/depth_buffer.cpp
    src/frame_pacer.cpp
    src/framebuffer.cpp
    src/gi_quality.cpp
    src/global_illumination.cpp
    src/golden_image.cpp
    src/ibl.cpp
    src/ibl_capture.cpp
    src/image_io.cpp
    src/lighting.cpp
    src/mesh.cpp
    src/pipeline.cpp
    src/pbr.cpp
    src/post_process.cpp
    src/post_process_quality.cpp
    src/present_target.cpp
    src/raster.cpp
    src/raytrace.cpp
    src/scratch_allocator.cpp
    src/reflections.cpp
    src/shader.cpp
    src/shadow.cpp
    src/shadow_quality.cpp
    src/sky.cpp
    src/temporal_aa.cpp
    src/texture.cpp
)

set(CYBER_SOFTWARE_HEADERS
    include/cyber/software/antialias.h
    include/cyber/software/camera.h
    include/cyber/software/clustered_pbr.h
    include/cyber/software/debug_text.h
    include/cyber/software/denoise.h
    include/cyber/software/depth_buffer.h
    include/cyber/software/frame_pacer.h
    include/cyber/software/framebuffer.h
    include/cyber/software/gi_quality.h
    include/cyber/software/global_illumination.h
    include/cyber/software/golden_image.h
    include/cyber/software/ibl.h
    include/cyber/software/image_io.h
    include/cyber/software/lighting.h
    include/cyber/software/mesh.h
    include/cyber/software/pipeline.h
    include/cyber/software/pbr.h
    include/cyber/software/post_process.h
    include/cyber/software/post_process_quality.h
    include/cyber/software/present_target.h
    include/cyber/software/raster.h
    include/cyber/software/raytrace.h
    include/cyber/software/shading.h
    include/cyber/software/shader.h
    include/cyber/software/shadow.h
    include/cyber/software/shadow_quality.h
    include/cyber/software/temporal_aa.h
    include/cyber/software/texture.h
    include/cyber/software/software.h
)

cyber_add_module(
    NAME software_renderer
    LAYER 4
    SOURCES ${CYBER_SOFTWARE_SOURCES}
    PUBLIC_HEADERS ${CYBER_SOFTWARE_HEADERS}
    DEPENDS core math memory platform jobs renderer
)

cyber_add_test(
    NAME software_framebuffer
    SOURCES tests/framebuffer_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_present
    SOURCES tests/present_target_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_image_io
    SOURCES tests/image_io_test.cpp
    DEPENDS software_renderer memory platform
)

cyber_add_test(
    NAME software_text
    SOURCES tests/debug_text_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_frame_pacer
    SOURCES tests/frame_pacer_test.cpp
    DEPENDS software_renderer platform
)

cyber_add_test(
    NAME software_screenshot
    SOURCES tests/screenshot_test.cpp
    DEPENDS software_renderer memory platform
)

cyber_add_test(
    NAME software_public_headers
    SOURCES tests/public_headers_test.cpp
    DEPENDS software_renderer
)

cyber_add_test(
    NAME software_raster
    SOURCES tests/raster_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_raster_coverage
    SOURCES tests/raster_coverage_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_depth_buffer
    SOURCES tests/depth_buffer_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_camera
    SOURCES tests/camera_test.cpp
    DEPENDS software_renderer
)

cyber_add_test(
    NAME software_pipeline
    SOURCES tests/pipeline_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_texture
    SOURCES tests/texture_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_shader
    SOURCES tests/shader_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_lighting
    SOURCES tests/lighting_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_pbr
    SOURCES tests/pbr_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_shadows
    SOURCES tests/shadow_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_shadow_acceptance
    SOURCES tests/shadow_acceptance_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_clustered_pbr
    SOURCES tests/clustered_pbr_test.cpp
    DEPENDS software_renderer renderer memory
)

add_executable(cyber_software_frame_bench benchmarks/frame_bench.cpp)
target_link_libraries(cyber_software_frame_bench
    PRIVATE cyber_software_renderer cyber_jobs cyber_memory cyber_math)
target_compile_features(cyber_software_frame_bench PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_software_frame_bench)

add_executable(cyber_software_clustered_pbr_bench benchmarks/clustered_pbr_bench.cpp)
target_link_libraries(cyber_software_clustered_pbr_bench
    PRIVATE cyber_software_renderer cyber_renderer cyber_memory)
target_compile_features(cyber_software_clustered_pbr_bench PRIVATE cxx_std_20)
cyber_apply_warnings(cyber_software_clustered_pbr_bench)

cyber_add_test(
    NAME software_image_load
    SOURCES tests/image_load_test.cpp
    DEPENDS software_renderer memory platform
)

cyber_add_test(
    NAME software_mesh
    SOURCES tests/mesh_test.cpp
    DEPENDS software_renderer memory platform
)

cyber_add_test(
    NAME software_golden_image
    SOURCES tests/golden_image_test.cpp
    DEPENDS software_renderer memory platform
)

cyber_add_test(
    NAME software_golden_scenes
    SOURCES tests/golden_scenes_test.cpp
    DEPENDS software_renderer memory platform
)

# Tests are skipped when a consumer adds this directory, so the target only
# exists in the CyberGraphics build.
if(TARGET cyber_test_software_golden_scenes)
    target_compile_definitions(
        cyber_test_software_golden_scenes
        PRIVATE CYBER_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/golden"
    )
endif()

cyber_add_test(
    NAME software_antialias
    SOURCES tests/antialias_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_ibl
    SOURCES tests/ibl_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_ibl_acceptance
    SOURCES tests/ibl_acceptance_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_post_process
    SOURCES tests/post_process_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_post_process_quality
    SOURCES tests/post_process_quality_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_temporal_aa
    SOURCES tests/temporal_aa_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_post_process_acceptance
    SOURCES tests/post_process_acceptance_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_raytrace
    SOURCES tests/raytrace_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_global_illumination
    SOURCES tests/global_illumination_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_denoise
    SOURCES tests/denoise_test.cpp
    DEPENDS software_renderer memory jobs
)

cyber_add_test(
    NAME software_gi_quality
    SOURCES tests/gi_quality_test.cpp
    DEPENDS software_renderer memory
)

cyber_add_test(
    NAME software_gi_acceptance
    SOURCES tests/gi_acceptance_test.cpp
    DEPENDS software_renderer memory jobs
)
