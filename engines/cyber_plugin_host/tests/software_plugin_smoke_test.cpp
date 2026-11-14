#include <cyber/plugin_host/renderer_plugin.h>
#include <cyber/test/test.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

template <typename T>
void Initialize(T& value)
{
    value = {};
    value.size = static_cast<cg_uint32>(sizeof(T));
    value.version = CG_STRUCT_VERSION_1;
}

}  // namespace

CG_TEST(software_plugin_renders_and_captures_a_triangle)
{
    auto loaded = cyber::plugin_host::RendererPlugin::Load(
        CYBER_TEST_PLUGIN_PATH,
        CG_CAPABILITY_SCENE_RENDERING | CG_CAPABILITY_SCREENSHOT | CG_CAPABILITY_RESIZE);
    CG_REQUIRE(loaded);
    const cg_plugin_api& api = loaded.Value().Api();

    cg_device_desc deviceDesc{};
    Initialize(deviceDesc);
    deviceDesc.width = 64;
    deviceDesc.height = 64;
    cg_device device = nullptr;
    CG_REQUIRE(api.create_device(&deviceDesc, &device) == CG_SUCCESS);

    cg_scene_desc sceneDesc{};
    Initialize(sceneDesc);
    cg_scene scene = nullptr;
    CG_REQUIRE(api.create_scene(device, &sceneDesc, &scene) == CG_SUCCESS);

    cg_view_desc viewDesc{};
    Initialize(viewDesc);
    for (cg_uint32 i = 0; i < 4; ++i)
    {
        viewDesc.view[i * 5] = 1.0f;
        viewDesc.projection[i * 5] = 1.0f;
    }

    viewDesc.viewport[2] = 64.0f;
    viewDesc.viewport[3] = 64.0f;
    viewDesc.clear_color[3] = 1.0f;
    cg_view view = nullptr;
    CG_REQUIRE(api.create_view(scene, &viewDesc, &view) == CG_SUCCESS);
    cg_view_desc invalidViewDesc = viewDesc;
    invalidViewDesc.view[0] = std::numeric_limits<float>::quiet_NaN();
    CG_EXPECT_EQ(api.update_view(view, &invalidViewDesc), CG_ERROR_INVALID_ARGUMENT);

    cg_vertex vertices[3]{};
    for (cg_vertex& vertex : vertices)
    {
        Initialize(vertex);
        vertex.color[0] = 1.0f;
        vertex.color[3] = 1.0f;
    }
    vertices[0].position[0] = -0.75f;
    vertices[0].position[1] = -0.75f;
    vertices[1].position[0] = 0.75f;
    vertices[1].position[1] = -0.75f;
    vertices[2].position[1] = 0.75f;
    for (cg_vertex& vertex : vertices)
    {
        vertex.position[2] = 0.5f;
    }
    const cg_uint32 indices[] = {0, 1, 2};
    cg_mesh_desc meshDesc{};
    Initialize(meshDesc);
    meshDesc.vertices = vertices;
    meshDesc.vertex_count = 3;
    meshDesc.indices = indices;
    meshDesc.index_count = 3;
    cg_mesh mesh = nullptr;
    const cg_uint32 invalidIndices[] = {0, 1, 3};
    meshDesc.indices = invalidIndices;
    CG_EXPECT_EQ(api.create_mesh(device, &meshDesc, &mesh), CG_ERROR_OUT_OF_RANGE);
    CG_EXPECT_EQ(mesh, nullptr);
    meshDesc.indices = indices;
    CG_REQUIRE(api.create_mesh(device, &meshDesc, &mesh) == CG_SUCCESS);

    cg_material_desc materialDesc{};
    Initialize(materialDesc);
    for (float& channel : materialDesc.base_color)
    {
        channel = 1.0f;
    }
    cg_material_desc invalidMaterialDesc = materialDesc;
    invalidMaterialDesc.base_color[0] = std::numeric_limits<float>::infinity();
    cg_material invalidMaterial = nullptr;
    CG_EXPECT_EQ(api.create_material(device, &invalidMaterialDesc, &invalidMaterial),
                 CG_ERROR_INVALID_ARGUMENT);
    CG_EXPECT_EQ(invalidMaterial, nullptr);
    cg_material material = nullptr;
    CG_REQUIRE(api.create_material(device, &materialDesc, &material) == CG_SUCCESS);

    cg_instance_desc instanceDesc{};
    Initialize(instanceDesc);
    instanceDesc.mesh = mesh;
    instanceDesc.material = material;
    for (cg_uint32 i = 0; i < 4; ++i)
    {
        instanceDesc.transform[i * 5] = 1.0f;
    }
    cg_instance instance = nullptr;
    cg_instance_desc invalidInstanceDesc = instanceDesc;
    invalidInstanceDesc.transform[0] = std::numeric_limits<float>::quiet_NaN();
    CG_EXPECT_EQ(api.create_instance(scene, &invalidInstanceDesc, &instance),
                 CG_ERROR_INVALID_ARGUMENT);
    CG_EXPECT_EQ(instance, nullptr);
    CG_REQUIRE(api.create_instance(scene, &instanceDesc, &instance) == CG_SUCCESS);
    cg_instance newerInstance = nullptr;
    CG_REQUIRE(api.create_instance(scene, &instanceDesc, &newerInstance) == CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_instance(scene, instance), CG_SUCCESS);
    CG_REQUIRE(api.submit_frame(device, scene, view) == CG_SUCCESS);

    cg_image image{};
    Initialize(image);
    image.format = CG_PIXEL_FORMAT_RGBA8_SRGB;
    std::vector<std::uint8_t> pixels(64 * 64 * 4);
    image.pixels = pixels.data();
    image.capacity_bytes = pixels.size();
    CG_REQUIRE(api.capture_screenshot(device, &image) == CG_SUCCESS);
    CG_EXPECT_EQ(image.width, 64U);
    CG_EXPECT_EQ(image.height, 64U);
    CG_EXPECT_EQ(image.required_bytes, pixels.size());

    bool foundRed = false;
    for (std::size_t offset = 0; offset < pixels.size(); offset += 4)
    {
        foundRed |= pixels[offset] > 200 && pixels[offset + 1] < 10 && pixels[offset + 2] < 10 &&
                    pixels[offset + 3] == 255;
    }
    CG_EXPECT_TRUE(foundRed);

    image.capacity_bytes = 4;
    CG_EXPECT_EQ(api.capture_screenshot(device, &image), CG_ERROR_BUFFER_TOO_SMALL);
    CG_EXPECT_EQ(image.required_bytes, pixels.size());
    CG_EXPECT_EQ(api.destroy_instance(scene, newerInstance), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_view(scene, view), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_scene(device, scene), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_mesh(device, mesh), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_material(device, material), CG_SUCCESS);
    CG_EXPECT_EQ(api.resize_device(device, 32, 24), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_device(device), CG_SUCCESS);
}

CG_TEST(software_plugin_accepts_the_original_device_descriptor_prefix)
{
    auto loaded = cyber::plugin_host::RendererPlugin::Load(CYBER_TEST_PLUGIN_PATH);
    CG_REQUIRE(loaded);
    cg_device_desc desc{};
    desc.size = static_cast<cg_uint32>(offsetof(cg_device_desc, native_window));
    desc.version = CG_STRUCT_VERSION_1;
    desc.width = 16;
    desc.height = 16;
    cg_device device = nullptr;
    CG_EXPECT_EQ(loaded.Value().Api().create_device(&desc, &device), CG_SUCCESS);
    CG_EXPECT_NE(device, nullptr);
    CG_EXPECT_EQ(loaded.Value().Api().destroy_device(device), CG_SUCCESS);
}
