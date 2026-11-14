#pragma once

#include <cyber/core/error.h>
#include <cyber/core/result.h>
#include <cyber/platform/library.h>

#include <cybergraphics/api.h>

namespace cyber::plugin_host
{

using cyber::core::Error;
using cyber::core::Result;

enum class RendererPreference
{
    CyberGraphics,
    Software,
};

struct RendererPaths
{
    const char* cyberGraphics = nullptr;
    const char* software = nullptr;
};

struct RendererLoadDiagnostics
{
    Error preferred{cyber::core::ErrorCode::Ok, nullptr};
    Error fallback{cyber::core::ErrorCode::Ok, nullptr};
    cg_result preferredDeviceResult = CG_SUCCESS;
    cg_result fallbackDeviceResult = CG_SUCCESS;
    bool fallbackAttempted = false;
};

class RendererPlugin
{
public:
    RendererPlugin() noexcept = default;
    RendererPlugin(const RendererPlugin&) = delete;
    RendererPlugin& operator=(const RendererPlugin&) = delete;
    RendererPlugin(RendererPlugin&&) noexcept = default;
    RendererPlugin& operator=(RendererPlugin&&) noexcept = default;

    [[nodiscard]] static Result<RendererPlugin, Error>
    Load(const char* path, cg_uint64 requiredCapabilities = 0) noexcept;

    [[nodiscard]] const cg_plugin_api& Api() const noexcept
    {
        return m_api;
    }

    [[nodiscard]] const cg_plugin_info& Info() const noexcept
    {
        return m_info;
    }

private:
    friend class RendererSelection;

    cyber::platform::DynamicLibrary m_library;
    cg_plugin_api m_api{};
    cg_plugin_info m_info{};
};

class RendererSelection
{
public:
    RendererSelection() noexcept = default;
    RendererSelection(const RendererSelection&) = delete;
    RendererSelection& operator=(const RendererSelection&) = delete;
    RendererSelection(RendererSelection&&) noexcept = default;
    RendererSelection& operator=(RendererSelection&&) noexcept = default;

    [[nodiscard]] static Result<RendererSelection, Error>
    Load(RendererPreference preference, const RendererPaths& paths, cg_uint64 requiredCapabilities,
         RendererLoadDiagnostics& diagnostics) noexcept;

    [[nodiscard]] RendererPreference Selected() const noexcept
    {
        return m_selected;
    }

    [[nodiscard]] const RendererPlugin& Plugin() const noexcept
    {
        return m_plugin;
    }

private:
    RendererPreference m_selected = RendererPreference::CyberGraphics;
    RendererPlugin m_plugin;
};

class RendererInstance
{
public:
    RendererInstance() noexcept = default;
    ~RendererInstance();
    RendererInstance(const RendererInstance&) = delete;
    RendererInstance& operator=(const RendererInstance&) = delete;
    RendererInstance(RendererInstance&& other) noexcept;
    RendererInstance& operator=(RendererInstance&& other) noexcept;

    [[nodiscard]] static Result<RendererInstance, Error>
    Create(RendererPreference preference, const RendererPaths& paths,
           cg_uint64 requiredCapabilities, const cg_device_desc& desc,
           RendererLoadDiagnostics& diagnostics) noexcept;

    [[nodiscard]] RendererPreference Selected() const noexcept
    {
        return m_selected;
    }

    [[nodiscard]] const RendererPlugin& Plugin() const noexcept
    {
        return m_plugin;
    }

    [[nodiscard]] cg_device Device() const noexcept
    {
        return m_device;
    }

private:
    void Release() noexcept;

    RendererPreference m_selected = RendererPreference::CyberGraphics;
    RendererPlugin m_plugin;
    cg_device m_device = nullptr;
};

}  // namespace cyber::plugin_host
