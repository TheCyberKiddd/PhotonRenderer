add_library(cyber_software_renderer_plugin SHARED
    ../engines/cyber_software_renderer/plugin/software_plugin.cpp
)
target_compile_features(cyber_software_renderer_plugin PRIVATE cxx_std_20)
target_compile_definitions(cyber_software_renderer_plugin PRIVATE CG_PLUGIN_BUILD)
target_link_libraries(cyber_software_renderer_plugin PRIVATE
    cyber::graphics_api
    cyber::software_renderer
)
cyber_apply_warnings(cyber_software_renderer_plugin)
if(CYBER_RUNTIME_OPTIONS)
    target_compile_options(cyber_software_renderer_plugin PRIVATE ${CYBER_RUNTIME_OPTIONS})
endif()
set_target_properties(cyber_software_renderer_plugin PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    OUTPUT_NAME cyber_software_renderer
)
install(TARGETS cyber_software_renderer_plugin
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(CYBER_RHI_ENABLE_D3D12 AND WIN32)
    add_library(cyber_d3d12_renderer_plugin SHARED d3d12_renderer_plugin.cpp)
    target_compile_features(cyber_d3d12_renderer_plugin PRIVATE cxx_std_20)
    target_compile_definitions(cyber_d3d12_renderer_plugin PRIVATE CG_PLUGIN_BUILD)
    target_link_libraries(cyber_d3d12_renderer_plugin PRIVATE
        cyber::graphics_api
        cyber::math
        cyber::memory
        cyber::rhi
    )
    cyber_apply_warnings(cyber_d3d12_renderer_plugin)
    if(CYBER_RUNTIME_OPTIONS)
        target_compile_options(cyber_d3d12_renderer_plugin PRIVATE ${CYBER_RUNTIME_OPTIONS})
    endif()
    set_target_properties(cyber_d3d12_renderer_plugin PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
        OUTPUT_NAME cyber_d3d12_renderer
    )
    install(TARGETS cyber_d3d12_renderer_plugin
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()
