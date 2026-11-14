cyber_add_module(
    NAME plugin_host
    LAYER 4
    SOURCES
        src/renderer_plugin.cpp
    PUBLIC_HEADERS
        include/cyber/plugin_host/renderer_plugin.h
    DEPENDS core platform cybergraphics_api
)

if(CYBER_BUILD_TESTS)
    add_library(cyber_test_renderer_plugin SHARED tests/mock_renderer_plugin.c)
    target_link_libraries(cyber_test_renderer_plugin PRIVATE cyber::cybergraphics_api)
    target_compile_definitions(cyber_test_renderer_plugin PRIVATE CG_PLUGIN_BUILD)
    set_target_properties(cyber_test_renderer_plugin PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
        OUTPUT_NAME cyber_test_renderer_plugin
    )

    add_library(cyber_test_renderer_plugin_legacy SHARED tests/mock_renderer_plugin.c)
    target_link_libraries(cyber_test_renderer_plugin_legacy PRIVATE cyber::cybergraphics_api)
    target_compile_definitions(cyber_test_renderer_plugin_legacy PRIVATE
        CG_PLUGIN_BUILD
        CG_TEST_LEGACY_TABLE
    )
    set_target_properties(cyber_test_renderer_plugin_legacy PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
        OUTPUT_NAME cyber_test_renderer_plugin_legacy
    )

    add_library(cyber_test_renderer_plugin_no_device SHARED tests/mock_renderer_plugin.c)
    target_link_libraries(cyber_test_renderer_plugin_no_device PRIVATE cyber::cybergraphics_api)
    target_compile_definitions(cyber_test_renderer_plugin_no_device PRIVATE
        CG_PLUGIN_BUILD
        CG_TEST_FAIL_DEVICE_CREATE
    )
    set_target_properties(cyber_test_renderer_plugin_no_device PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
        OUTPUT_NAME cyber_test_renderer_plugin_no_device
    )

    cyber_add_test(
        NAME renderer_plugin_runtime
        SOURCES tests/renderer_plugin_runtime_test.cpp
        DEPENDS plugin_host
    )
    target_compile_definitions(cyber_test_renderer_plugin_runtime PRIVATE
        CYBER_TEST_PLUGIN_PATH="$<TARGET_FILE:cyber_test_renderer_plugin>"
    )
    add_dependencies(cyber_test_renderer_plugin_runtime cyber_test_renderer_plugin)
    target_compile_definitions(cyber_test_renderer_plugin_runtime PRIVATE
        CYBER_TEST_LEGACY_PLUGIN_PATH="$<TARGET_FILE:cyber_test_renderer_plugin_legacy>"
    )
    add_dependencies(cyber_test_renderer_plugin_runtime cyber_test_renderer_plugin_legacy)
    target_compile_definitions(cyber_test_renderer_plugin_runtime PRIVATE
        CYBER_TEST_NO_DEVICE_PLUGIN_PATH="$<TARGET_FILE:cyber_test_renderer_plugin_no_device>"
    )
    add_dependencies(cyber_test_renderer_plugin_runtime cyber_test_renderer_plugin_no_device)

    if(CYBER_BUILD_PLUGINS)
        cyber_add_test(
            NAME software_renderer_plugin_smoke
            SOURCES tests/software_plugin_smoke_test.cpp
            DEPENDS plugin_host
        )
        target_compile_definitions(cyber_test_software_renderer_plugin_smoke PRIVATE
            CYBER_TEST_PLUGIN_PATH="$<TARGET_FILE:cyber_software_renderer_plugin>"
        )
        add_dependencies(cyber_test_software_renderer_plugin_smoke
            cyber_software_renderer_plugin
        )

        if(CYBER_RHI_ENABLE_D3D12 AND WIN32)
            cyber_add_test(
                NAME d3d12_renderer_plugin_abi
                SOURCES tests/d3d12_plugin_abi_test.cpp
                DEPENDS plugin_host platform memory rhi
            )
            target_compile_definitions(cyber_test_d3d12_renderer_plugin_abi PRIVATE
                CYBER_TEST_D3D12_PLUGIN_PATH="$<TARGET_FILE:cyber_d3d12_renderer_plugin>"
                CYBER_TEST_SOFTWARE_PLUGIN_PATH="$<TARGET_FILE:cyber_software_renderer_plugin>"
            )
            add_dependencies(cyber_test_d3d12_renderer_plugin_abi
                cyber_d3d12_renderer_plugin
                cyber_software_renderer_plugin
            )
        endif()
    endif()
endif()
