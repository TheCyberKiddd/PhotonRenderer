#include <cyber/plugin_host/renderer_plugin.h>

#include <cstddef>
#include <cstring>
#include <utility>

namespace cyber::plugin_host
{
namespace
{

constexpr const char* kMissingPath = "renderer plugin path is empty";
constexpr const char* kMissingEntry = "renderer plugin entry point is missing";
constexpr const char* kBadTable = "renderer plugin returned an invalid API table";
constexpr const char* kBadInfo = "renderer plugin returned invalid plugin information";
constexpr const char* kMissingCapability = "renderer plugin lacks a required capability";
constexpr const char* kInvalidPreference = "renderer preference is invalid";
constexpr const char* kDeviceCreationFailed = "preferred renderer device creation failed";
constexpr const char* kFallbackDeviceCreationFailed = "fallback renderer device creation failed";

cyber::core::ErrorCode ToCoreError(cg_result result) noexcept
{
    switch (result)
    {
    case CG_ERROR_INVALID_ARGUMENT:
        return cyber::core::ErrorCode::InvalidArgument;
    case CG_ERROR_OUT_OF_RANGE:
        return cyber::core::ErrorCode::OutOfRange;
    case CG_ERROR_NOT_FOUND:
        return cyber::core::ErrorCode::NotFound;
    case CG_ERROR_UNSUPPORTED:
        return cyber::core::ErrorCode::Unsupported;
    case CG_ERROR_LIMIT_EXCEEDED:
        return cyber::core::ErrorCode::Overflow;
    case CG_ERROR_INVALID_STATE:
        return cyber::core::ErrorCode::Busy;
    case CG_SUCCESS:
        return cyber::core::ErrorCode::Ok;
    case CG_ERROR_INCOMPATIBLE_VERSION:
    case CG_ERROR_OUT_OF_MEMORY:
    case CG_ERROR_BUFFER_TOO_SMALL:
    case CG_ERROR_INTERNAL:
    default:
        return cyber::core::ErrorCode::Internal;
    }
}

template <typename T>
bool HasTableField(const cg_plugin_api& api, std::size_t offset, const T&) noexcept
{
    return static_cast<std::size_t>(api.size) >= offset + sizeof(T);
}

bool HasRequiredFunctions(const cg_plugin_api& api, cg_uint64 capabilities) noexcept
{
    if (!HasTableField(api, offsetof(cg_plugin_api, get_info), api.get_info) ||
        !HasTableField(api, offsetof(cg_plugin_api, create_device), api.create_device) ||
        !HasTableField(api, offsetof(cg_plugin_api, destroy_device), api.destroy_device) ||
        api.get_info == nullptr || api.create_device == nullptr || api.destroy_device == nullptr)
    {
        return false;
    }
    if ((capabilities & CG_CAPABILITY_RESIZE) != 0 &&
        (!HasTableField(api, offsetof(cg_plugin_api, resize_device), api.resize_device) ||
         api.resize_device == nullptr))
    {
        return false;
    }
    if ((capabilities & CG_CAPABILITY_SCENE_RENDERING) != 0 &&
        (!HasTableField(api, offsetof(cg_plugin_api, create_scene), api.create_scene) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_scene), api.destroy_scene) ||
         !HasTableField(api, offsetof(cg_plugin_api, create_view), api.create_view) ||
         !HasTableField(api, offsetof(cg_plugin_api, update_view), api.update_view) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_view), api.destroy_view) ||
         !HasTableField(api, offsetof(cg_plugin_api, create_mesh), api.create_mesh) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_mesh), api.destroy_mesh) ||
         !HasTableField(api, offsetof(cg_plugin_api, create_material), api.create_material) ||
         !HasTableField(api, offsetof(cg_plugin_api, update_material), api.update_material) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_material), api.destroy_material) ||
         !HasTableField(api, offsetof(cg_plugin_api, create_instance), api.create_instance) ||
         !HasTableField(api, offsetof(cg_plugin_api, update_instance), api.update_instance) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_instance), api.destroy_instance) ||
         !HasTableField(api, offsetof(cg_plugin_api, submit_frame), api.submit_frame) ||
         api.create_scene == nullptr || api.destroy_scene == nullptr ||
         api.create_view == nullptr || api.update_view == nullptr || api.destroy_view == nullptr ||
         api.create_mesh == nullptr || api.destroy_mesh == nullptr ||
         api.create_material == nullptr || api.update_material == nullptr ||
         api.destroy_material == nullptr || api.create_instance == nullptr ||
         api.update_instance == nullptr || api.destroy_instance == nullptr ||
         api.submit_frame == nullptr))
    {
        return false;
    }
    if ((capabilities & CG_CAPABILITY_LIGHTING) != 0 &&
        (!HasTableField(api, offsetof(cg_plugin_api, create_light), api.create_light) ||
         !HasTableField(api, offsetof(cg_plugin_api, update_light), api.update_light) ||
         !HasTableField(api, offsetof(cg_plugin_api, destroy_light), api.destroy_light) ||
         api.create_light == nullptr || api.update_light == nullptr ||
         api.destroy_light == nullptr))
    {
        return false;
    }
    if ((capabilities & CG_CAPABILITY_DEBUG_LINES) != 0 &&
        (!HasTableField(api, offsetof(cg_plugin_api, draw_debug_lines), api.draw_debug_lines) ||
         api.draw_debug_lines == nullptr))
    {
        return false;
    }
    if ((capabilities & CG_CAPABILITY_SCREENSHOT) != 0 &&
        (!HasTableField(api, offsetof(cg_plugin_api, capture_screenshot), api.capture_screenshot) ||
         api.capture_screenshot == nullptr))
    {
        return false;
    }
    return true;
}

}  // namespace

Result<RendererPlugin, Error> RendererPlugin::Load(const char* path,
                                                   cg_uint64 requiredCapabilities) noexcept
{
    if (path == nullptr || path[0] == '\0')
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument, kMissingPath});
    }

    auto opened = cyber::platform::DynamicLibrary::Open(path);
    if (!opened)
    {
        return Result<RendererPlugin, Error>::Err(opened.Error());
    }

    RendererPlugin plugin;
    plugin.m_library = std::move(opened.Value());
    auto entry = plugin.m_library.GetFunction<cg_get_plugin_api_fn>(CG_PLUGIN_ENTRY_POINT);
    if (!entry)
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::NotFound, kMissingEntry});
    }

    plugin.m_api = {};
    plugin.m_api.size = static_cast<cg_uint32>(sizeof(plugin.m_api));
    plugin.m_api.version = CG_STRUCT_VERSION_1;
    const cg_result negotiated = entry.Value()(CG_ABI_VERSION, &plugin.m_api);
    if (negotiated != CG_SUCCESS ||
        plugin.m_api.size < offsetof(cg_plugin_api, get_info) + sizeof(plugin.m_api.get_info) ||
        plugin.m_api.size > sizeof(plugin.m_api) || plugin.m_api.version != CG_STRUCT_VERSION_1 ||
        plugin.m_api.abi_version != CG_ABI_VERSION ||
        !HasTableField(plugin.m_api, offsetof(cg_plugin_api, get_info), plugin.m_api.get_info) ||
        plugin.m_api.get_info == nullptr)
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::Unsupported, kBadTable});
    }

    plugin.m_info = {};
    plugin.m_info.size = static_cast<cg_uint32>(sizeof(plugin.m_info));
    plugin.m_info.version = CG_STRUCT_VERSION_1;
    const cg_result infoResult = plugin.m_api.get_info(&plugin.m_info);
    if (infoResult != CG_SUCCESS ||
        plugin.m_info.size <
            offsetof(cg_plugin_info, capabilities) + sizeof(plugin.m_info.capabilities) ||
        plugin.m_info.size > sizeof(plugin.m_info) ||
        plugin.m_info.version != CG_STRUCT_VERSION_1 || plugin.m_info.name == nullptr)
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument, kBadInfo});
    }
    if ((plugin.m_info.capabilities & requiredCapabilities) != requiredCapabilities)
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::Unsupported, kMissingCapability});
    }
    if (!HasRequiredFunctions(plugin.m_api, requiredCapabilities))
    {
        return Result<RendererPlugin, Error>::Err(
            Error{cyber::core::ErrorCode::Unsupported, kBadTable});
    }

    return Result<RendererPlugin, Error>::Ok(std::move(plugin));
}

Result<RendererSelection, Error>
RendererSelection::Load(RendererPreference preference, const RendererPaths& paths,
                        cg_uint64 requiredCapabilities,
                        RendererLoadDiagnostics& diagnostics) noexcept
{
    diagnostics = {};
    if (preference != RendererPreference::CyberGraphics &&
        preference != RendererPreference::Software)
    {
        return Result<RendererSelection, Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument, kInvalidPreference});
    }
    const char* preferredPath =
        preference == RendererPreference::CyberGraphics ? paths.cyberGraphics : paths.software;
    const char* fallbackPath =
        preference == RendererPreference::CyberGraphics ? paths.software : paths.cyberGraphics;
    const RendererPreference fallbackPreference = preference == RendererPreference::CyberGraphics
                                                      ? RendererPreference::Software
                                                      : RendererPreference::CyberGraphics;

    auto preferred = RendererPlugin::Load(preferredPath, requiredCapabilities);
    if (preferred)
    {
        RendererSelection selection;
        selection.m_selected = preference;
        selection.m_plugin = std::move(preferred.Value());
        return Result<RendererSelection, Error>::Ok(std::move(selection));
    }
    diagnostics.preferred = preferred.Error();

    diagnostics.fallbackAttempted = true;
    auto fallback = RendererPlugin::Load(fallbackPath, requiredCapabilities);
    if (!fallback)
    {
        diagnostics.fallback = fallback.Error();
        return Result<RendererSelection, Error>::Err(diagnostics.fallback);
    }

    RendererSelection selection;
    selection.m_selected = fallbackPreference;
    selection.m_plugin = std::move(fallback.Value());
    return Result<RendererSelection, Error>::Ok(std::move(selection));
}

RendererInstance::~RendererInstance()
{
    Release();
}

RendererInstance::RendererInstance(RendererInstance&& other) noexcept
    : m_selected(other.m_selected), m_plugin(std::move(other.m_plugin)), m_device(other.m_device)
{
    other.m_device = nullptr;
}

RendererInstance& RendererInstance::operator=(RendererInstance&& other) noexcept
{
    if (this != &other)
    {
        Release();
        m_selected = other.m_selected;
        m_plugin = std::move(other.m_plugin);
        m_device = other.m_device;
        other.m_device = nullptr;
    }
    return *this;
}

void RendererInstance::Release() noexcept
{
    if (m_device != nullptr)
    {
        (void)m_plugin.Api().destroy_device(m_device);
        m_device = nullptr;
    }
}

Result<RendererInstance, Error>
RendererInstance::Create(RendererPreference preference, const RendererPaths& paths,
                         cg_uint64 requiredCapabilities, const cg_device_desc& desc,
                         RendererLoadDiagnostics& diagnostics) noexcept
{
    diagnostics = {};
    if (preference != RendererPreference::CyberGraphics &&
        preference != RendererPreference::Software)
    {
        return Result<RendererInstance, Error>::Err(
            Error{cyber::core::ErrorCode::InvalidArgument, kInvalidPreference});
    }

    const char* preferredPath =
        preference == RendererPreference::CyberGraphics ? paths.cyberGraphics : paths.software;
    const char* fallbackPath =
        preference == RendererPreference::CyberGraphics ? paths.software : paths.cyberGraphics;
    const RendererPreference fallbackPreference = preference == RendererPreference::CyberGraphics
                                                      ? RendererPreference::Software
                                                      : RendererPreference::CyberGraphics;

    auto preferred = RendererPlugin::Load(preferredPath, requiredCapabilities);
    if (!preferred)
    {
        diagnostics.preferred = preferred.Error();
    }
    else
    {
        RendererInstance instance;
        instance.m_selected = preference;
        instance.m_plugin = std::move(preferred.Value());
        const cg_result created = instance.m_plugin.Api().create_device(&desc, &instance.m_device);
        const cg_result deviceResult =
            created == CG_SUCCESS && instance.m_device == nullptr ? CG_ERROR_INTERNAL : created;
        if (deviceResult == CG_SUCCESS)
        {
            return Result<RendererInstance, Error>::Ok(std::move(instance));
        }
        if (instance.m_device != nullptr)
        {
            (void)instance.m_plugin.Api().destroy_device(instance.m_device);
            instance.m_device = nullptr;
        }
        diagnostics.preferredDeviceResult = deviceResult;
        diagnostics.preferred = Error{ToCoreError(deviceResult), kDeviceCreationFailed};
    }

    diagnostics.fallbackAttempted = true;
    auto fallback = RendererPlugin::Load(fallbackPath, requiredCapabilities);
    if (!fallback)
    {
        diagnostics.fallback = fallback.Error();
        return Result<RendererInstance, Error>::Err(diagnostics.fallback);
    }

    RendererInstance instance;
    instance.m_selected = fallbackPreference;
    instance.m_plugin = std::move(fallback.Value());
    const cg_result created = instance.m_plugin.Api().create_device(&desc, &instance.m_device);
    const cg_result deviceResult =
        created == CG_SUCCESS && instance.m_device == nullptr ? CG_ERROR_INTERNAL : created;
    if (deviceResult != CG_SUCCESS)
    {
        if (instance.m_device != nullptr)
        {
            (void)instance.m_plugin.Api().destroy_device(instance.m_device);
            instance.m_device = nullptr;
        }
        diagnostics.fallbackDeviceResult = deviceResult;
        diagnostics.fallback = Error{ToCoreError(deviceResult), kFallbackDeviceCreationFailed};
        return Result<RendererInstance, Error>::Err(diagnostics.fallback);
    }
    return Result<RendererInstance, Error>::Ok(std::move(instance));
}

}  // namespace cyber::plugin_host
