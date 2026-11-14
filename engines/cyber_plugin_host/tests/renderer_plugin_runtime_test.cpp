#include <cyber/plugin_host/renderer_plugin.h>
#include <cyber/test/test.h>

namespace
{

constexpr cg_uint64 kRequired = CG_CAPABILITY_SCREENSHOT | CG_CAPABILITY_RESIZE;

}  // namespace

CG_TEST(renderer_plugin_loads_and_negotiates_capabilities)
{
    auto plugin = cyber::plugin_host::RendererPlugin::Load(CYBER_TEST_PLUGIN_PATH, kRequired);
    CG_REQUIRE(plugin);
    CG_EXPECT_STREQ(plugin.Value().Info().name, "Plugin Host Test Renderer");
    CG_EXPECT_EQ(plugin.Value().Api().resize_device(nullptr, 1, 1), CG_ERROR_INVALID_ARGUMENT);

    auto missingCapability = cyber::plugin_host::RendererPlugin::Load(
        CYBER_TEST_PLUGIN_PATH, CG_CAPABILITY_SCENE_RENDERING);
    CG_EXPECT_FALSE(missingCapability);
    CG_EXPECT_EQ(missingCapability.Error().code, cyber::core::ErrorCode::Unsupported);
}

CG_TEST(renderer_plugin_accepts_a_compatible_older_function_table)
{
    auto plugin = cyber::plugin_host::RendererPlugin::Load(CYBER_TEST_LEGACY_PLUGIN_PATH,
                                                           CG_CAPABILITY_RESIZE);
    CG_REQUIRE(plugin);
    CG_EXPECT_EQ(plugin.Value().Api().size, offsetof(cg_plugin_api, create_scene));
    CG_EXPECT_EQ(plugin.Value().Api().resize_device(nullptr, 1, 1), CG_ERROR_INVALID_ARGUMENT);
}

CG_TEST(renderer_plugin_fallback_preserves_preference_failure)
{
    cyber::plugin_host::RendererPaths paths{};
    paths.cyberGraphics = "cybergraphics_plugin_does_not_exist";
    paths.software = CYBER_TEST_PLUGIN_PATH;
    cyber::plugin_host::RendererLoadDiagnostics diagnostics{};
    auto selected = cyber::plugin_host::RendererSelection::Load(
        cyber::plugin_host::RendererPreference::CyberGraphics, paths, kRequired, diagnostics);
    CG_REQUIRE(selected);
    CG_EXPECT_EQ(selected.Value().Selected(), cyber::plugin_host::RendererPreference::Software);
    CG_EXPECT_TRUE(diagnostics.fallbackAttempted);
    CG_EXPECT_NE(diagnostics.preferred.code, cyber::core::ErrorCode::Ok);

    cg_device_desc desc{};
    desc.size = sizeof(desc);
    desc.version = CG_STRUCT_VERSION_1;
    cg_device device = nullptr;
    const cg_plugin_api& api = selected.Value().Plugin().Api();
    CG_REQUIRE(api.create_device(&desc, &device) == CG_SUCCESS);
    CG_EXPECT_NE(device, nullptr);
    CG_EXPECT_EQ(api.resize_device(device, 640, 360), CG_SUCCESS);
    CG_EXPECT_EQ(api.destroy_device(device), CG_SUCCESS);
}

CG_TEST(renderer_plugin_reports_both_load_failures)
{
    cyber::plugin_host::RendererPaths paths{};
    paths.cyberGraphics = "cybergraphics_plugin_does_not_exist";
    paths.software = "cybergraphics_fallback_does_not_exist";
    cyber::plugin_host::RendererLoadDiagnostics diagnostics{};
    auto selected = cyber::plugin_host::RendererSelection::Load(
        cyber::plugin_host::RendererPreference::CyberGraphics, paths, kRequired, diagnostics);
    CG_EXPECT_FALSE(selected);
    CG_EXPECT_TRUE(diagnostics.fallbackAttempted);
    CG_EXPECT_NE(diagnostics.preferred.code, cyber::core::ErrorCode::Ok);
    CG_EXPECT_NE(diagnostics.fallback.code, cyber::core::ErrorCode::Ok);
}

CG_TEST(renderer_plugin_rejects_an_invalid_preference)
{
    cyber::plugin_host::RendererPaths paths{};
    cyber::plugin_host::RendererLoadDiagnostics diagnostics{};
    auto selected = cyber::plugin_host::RendererSelection::Load(
        static_cast<cyber::plugin_host::RendererPreference>(42), paths, kRequired, diagnostics);
    CG_EXPECT_FALSE(selected);
    CG_EXPECT_EQ(selected.Error().code, cyber::core::ErrorCode::InvalidArgument);
    CG_EXPECT_FALSE(diagnostics.fallbackAttempted);
}

CG_TEST(renderer_plugin_falls_back_when_preferred_device_creation_fails)
{
    cyber::plugin_host::RendererPaths paths{};
    paths.cyberGraphics = CYBER_TEST_NO_DEVICE_PLUGIN_PATH;
    paths.software = CYBER_TEST_PLUGIN_PATH;
    cyber::plugin_host::RendererLoadDiagnostics diagnostics{};
    cg_device_desc desc{};
    desc.size = sizeof(desc);
    desc.version = CG_STRUCT_VERSION_1;
    desc.width = 64;
    desc.height = 64;
    auto instance = cyber::plugin_host::RendererInstance::Create(
        cyber::plugin_host::RendererPreference::CyberGraphics, paths, kRequired, desc, diagnostics);
    CG_REQUIRE(instance);
    CG_EXPECT_EQ(instance.Value().Selected(), cyber::plugin_host::RendererPreference::Software);
    CG_EXPECT_NE(instance.Value().Device(), nullptr);
    CG_EXPECT_TRUE(diagnostics.fallbackAttempted);
    CG_EXPECT_NE(diagnostics.preferred.code, cyber::core::ErrorCode::Ok);
    CG_EXPECT_EQ(diagnostics.fallback.code, cyber::core::ErrorCode::Ok);
}
