#include <cyber/memory/heap_allocator.h>
#include <cyber/platform/window.h>
#include <cyber/plugin_host/renderer_plugin.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/test/test.h>

#include <cstdio>

namespace
{

cg_uint32 g_callbackCount = 0;
cg_result g_callbackResult = CG_SUCCESS;
const char* g_callbackMessage = nullptr;

void CG_CALL ErrorCallback(void*, cg_result result, const char* message)
{
    ++g_callbackCount;
    g_callbackResult = result;
    g_callbackMessage = message;
}

}  // namespace

CG_TEST(d3d12_plugin_exports_scene_renderer_and_reports_surface_errors)
{
    auto plugin = cyber::plugin_host::RendererPlugin::Load(
        CYBER_TEST_D3D12_PLUGIN_PATH, CG_CAPABILITY_SCENE_RENDERING | CG_CAPABILITY_RESIZE);
    CG_REQUIRE(plugin);
    CG_EXPECT_STREQ(plugin.Value().Info().name, "CyberGraphics D3D12 Renderer");

    g_callbackCount = 0;
    g_callbackResult = CG_SUCCESS;
    g_callbackMessage = nullptr;
    cg_device_desc desc{};
    desc.size = sizeof(desc);
    desc.version = CG_STRUCT_VERSION_1;
    desc.width = 640;
    desc.height = 360;
    desc.error_callback = ErrorCallback;
    cg_device device = nullptr;
    CG_EXPECT_EQ(plugin.Value().Api().create_device(&desc, &device), CG_ERROR_UNSUPPORTED);
    CG_EXPECT_EQ(device, nullptr);
    CG_EXPECT_EQ(g_callbackCount, 1U);
    CG_EXPECT_EQ(g_callbackResult, CG_ERROR_UNSUPPORTED);

    cyber::plugin_host::RendererPaths paths{};
    paths.cyberGraphics = CYBER_TEST_D3D12_PLUGIN_PATH;
    paths.software = CYBER_TEST_SOFTWARE_PLUGIN_PATH;
    cyber::plugin_host::RendererLoadDiagnostics diagnostics{};
    auto selected = cyber::plugin_host::RendererInstance::Create(
        cyber::plugin_host::RendererPreference::CyberGraphics, paths,
        CG_CAPABILITY_SCENE_RENDERING | CG_CAPABILITY_RESIZE, desc, diagnostics);
    CG_REQUIRE(selected);
    CG_EXPECT_EQ(selected.Value().Selected(), cyber::plugin_host::RendererPreference::Software);
    CG_EXPECT_TRUE(diagnostics.fallbackAttempted);
    CG_EXPECT_EQ(diagnostics.preferredDeviceResult, CG_ERROR_UNSUPPORTED);

    const cyber::rhi::Backend* backend = cyber::rhi::D3D12Backend();
    CG_REQUIRE(backend != nullptr);
    if (backend->enumerateAdapters(nullptr, 0) == 0)
    {
        CG_SKIP("no Direct3D 12 adapter is available");
    }

    cyber::memory::HeapAllocator allocator;
    cyber::platform::WindowDesc windowDesc{};
    windowDesc.title = "CyberGraphics D3D12 plugin test";
    windowDesc.width = 64;
    windowDesc.height = 64;
    windowDesc.visible = false;
    auto window = cyber::platform::Window::Create(windowDesc, allocator);
    CG_REQUIRE(window);
    desc.width = static_cast<cg_uint32>(window.Value().Width());
    desc.height = static_cast<cg_uint32>(window.Value().Height());
    desc.native_window = window.Value().NativeHandle();
    const cg_result createResult = plugin.Value().Api().create_device(&desc, &device);
    if (createResult != CG_SUCCESS)
    {
        std::printf("D3D12 plugin device creation failed (%d): %s\n", createResult,
                    g_callbackMessage != nullptr ? g_callbackMessage : "no callback message");
    }
    CG_REQUIRE(createResult == CG_SUCCESS);

    const cg_plugin_api& api = plugin.Value().Api();
    cg_scene_desc sceneDesc{};
    sceneDesc.size = sizeof(sceneDesc);
    sceneDesc.version = CG_STRUCT_VERSION_1;
    cg_scene scene = nullptr;
    CG_REQUIRE(api.create_scene(device, &sceneDesc, &scene) == CG_SUCCESS);

    cg_view_desc viewDesc{};
    viewDesc.size = sizeof(viewDesc);
    viewDesc.version = CG_STRUCT_VERSION_1;
    for (cg_uint32 index = 0; index < 4; ++index)
    {
        viewDesc.view[index * 5] = 1.0f;
        viewDesc.projection[index * 5] = 1.0f;
    }
    viewDesc.viewport[2] = static_cast<float>(desc.width);
    viewDesc.viewport[3] = static_cast<float>(desc.height);
    viewDesc.clear_color[3] = 1.0f;
    cg_view view = nullptr;
    CG_REQUIRE(api.create_view(scene, &viewDesc, &view) == CG_SUCCESS);

    cg_vertex vertices[3]{};
    for (cg_vertex& vertex : vertices)
    {
        vertex.size = sizeof(vertex);
        vertex.version = CG_STRUCT_VERSION_1;
        vertex.position[2] = 0.5f;
        vertex.color[0] = 1.0f;
        vertex.color[3] = 1.0f;
    }
    vertices[0].position[0] = -0.75f;
    vertices[0].position[1] = -0.75f;
    vertices[1].position[0] = 0.75f;
    vertices[1].position[1] = -0.75f;
    vertices[2].position[1] = 0.75f;
    const cg_uint32 indices[] = {0, 1, 2};
    cg_mesh_desc meshDesc{};
    meshDesc.size = sizeof(meshDesc);
    meshDesc.version = CG_STRUCT_VERSION_1;
    meshDesc.vertices = vertices;
    meshDesc.vertex_count = 3;
    meshDesc.indices = indices;
    meshDesc.index_count = 3;
    cg_mesh mesh = nullptr;
    CG_REQUIRE(api.create_mesh(device, &meshDesc, &mesh) == CG_SUCCESS);

    cg_material_desc materialDesc{};
    materialDesc.size = sizeof(materialDesc);
    materialDesc.version = CG_STRUCT_VERSION_1;
    materialDesc.base_color[0] = 1.0f;
    materialDesc.base_color[1] = 1.0f;
    materialDesc.base_color[2] = 1.0f;
    materialDesc.base_color[3] = 1.0f;
    cg_material material = nullptr;
    CG_REQUIRE(api.create_material(device, &materialDesc, &material) == CG_SUCCESS);

    cg_instance_desc instanceDesc{};
    instanceDesc.size = sizeof(instanceDesc);
    instanceDesc.version = CG_STRUCT_VERSION_1;
    instanceDesc.mesh = mesh;
    instanceDesc.material = material;
    for (cg_uint32 index = 0; index < 4; ++index)
    {
        instanceDesc.transform[index * 5] = 1.0f;
    }
    cg_instance instance = nullptr;
    CG_REQUIRE(api.create_instance(scene, &instanceDesc, &instance) == CG_SUCCESS);
    CG_EXPECT_EQ(api.submit_frame(device, scene, view), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_instance(scene, instance), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_view(scene, view), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_scene(device, scene), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_mesh(device, mesh), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_material(device, material), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_device(device), CG_SUCCESS);
}
