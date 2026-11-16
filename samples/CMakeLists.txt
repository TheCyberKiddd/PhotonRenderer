# Runnable samples. Every sample compiles in CI; a sample that needs a window
# still builds on platforms whose backend returns Unsupported at runtime.

add_subdirectory(00_window)
add_subdirectory(01_gradient)
add_subdirectory(02_raster)
add_subdirectory(03_depth)
add_subdirectory(04_textured)
add_subdirectory(04_lighting)
add_subdirectory(11_reflections)
add_subdirectory(10_antialiasing)
add_subdirectory(12_post_process)
add_subdirectory(13_showcase)
add_subdirectory(14_global_illumination)
add_subdirectory(15_gallery)
add_subdirectory(18_texture_streaming)
add_subdirectory(19_virtual_texture)

# The RHI samples run the same code on D3D12 and Vulkan (`--backend=`), so they
# build whenever either backend is in the build.
if(CYBER_RHI_ENABLE_D3D12 OR CYBER_RHI_ENABLE_VULKAN)
    add_subdirectory(05_d3d12_markers)
    add_subdirectory(06_shader_hot_reload)
    add_subdirectory(07_materials)
    add_subdirectory(08_frame_streaming)
    add_subdirectory(09_clustered_lighting)
    add_subdirectory(17_gpu_driven)
endif()

# Phase 34: every RHI sample runs once per compiled-in backend. A sample exits
# with 77 when the machine has no adapter (or no window system) for a backend,
# which CTest reports as skipped rather than failed.
function(cyber_add_sample_runs name target)
    if(NOT CYBER_BUILD_TESTS OR NOT TARGET ${target})
        return()
    endif()
    set(backends)
    if(CYBER_RHI_ENABLE_D3D12)
        list(APPEND backends d3d12)
    endif()
    if(CYBER_RHI_ENABLE_VULKAN)
        list(APPEND backends vulkan)
    endif()
    foreach(backend IN LISTS backends)
        add_test(NAME sample_${name}_${backend}
                 COMMAND ${target} ${ARGN} --backend=${backend}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
        set_tests_properties(sample_${name}_${backend} PROPERTIES
            SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE LABELS "sample;${backend}")
    endforeach()
endfunction()

cyber_add_sample_runs(gpu_markers cyber_sample_d3d12_markers --frames=60)
cyber_add_sample_runs(shader_hot_reload cyber_sample_shader_hot_reload --self-test)
cyber_add_sample_runs(materials cyber_sample_materials --self-test)
cyber_add_sample_runs(frame_streaming cyber_sample_frame_streaming)
cyber_add_sample_runs(clustered_lighting cyber_sample_clustered_lighting)
cyber_add_sample_runs(global_illumination cyber_sample_global_illumination
                      ${CMAKE_CURRENT_BINARY_DIR}/global_illumination_test.tga)
cyber_add_sample_runs(gpu_driven cyber_sample_gpu_driven)
