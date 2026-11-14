#include <cybergraphics/api.h>

#include <stddef.h>
#include <string.h>

static cg_result CG_CALL get_info(cg_plugin_info* info)
{
    if (info == NULL || info->size < sizeof(*info) || info->version != CG_STRUCT_VERSION_1)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    info->name = "Plugin Host Test Renderer";
#if defined(CG_TEST_LEGACY_TABLE)
    info->capabilities = CG_CAPABILITY_RESIZE;
#else
    info->capabilities = CG_CAPABILITY_SCREENSHOT | CG_CAPABILITY_RESIZE;
#endif
    return CG_SUCCESS;
}

static cg_result CG_CALL create_device(const cg_device_desc* desc, cg_device* out_device)
{
    if (desc == NULL || desc->size < offsetof(cg_device_desc, native_window) ||
        desc->version != CG_STRUCT_VERSION_1 ||
        out_device == NULL)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
#if defined(CG_TEST_FAIL_DEVICE_CREATE)
    *out_device = NULL;
    return CG_ERROR_UNSUPPORTED;
#else
    *out_device = (cg_device)(uintptr_t)1;
    return CG_SUCCESS;
#endif
}

static cg_result CG_CALL destroy_device(cg_device device)
{
    return device == (cg_device)(uintptr_t)1 ? CG_SUCCESS : CG_ERROR_INVALID_ARGUMENT;
}

static cg_result CG_CALL resize_device(cg_device device, cg_uint32 width, cg_uint32 height)
{
    return device == (cg_device)(uintptr_t)1 && width > 0 && height > 0
               ? CG_SUCCESS
               : CG_ERROR_INVALID_ARGUMENT;
}

static cg_result CG_CALL capture_screenshot(cg_device device, cg_image* image)
{
    (void)device;
    (void)image;
    return CG_ERROR_UNSUPPORTED;
}

CG_PLUGIN_EXPORT cg_result CG_CALL cgGetPluginApi(cg_uint32 requested_abi_version,
                                                   cg_plugin_api* in_out_api)
{
    if (in_out_api == NULL ||
        in_out_api->size < offsetof(cg_plugin_api, get_info) + sizeof(in_out_api->get_info))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (requested_abi_version != CG_ABI_VERSION)
    {
        return CG_ERROR_INCOMPATIBLE_VERSION;
    }

    cg_plugin_api api;
    memset(&api, 0, sizeof(api));
    api.size = (cg_uint32)sizeof(api);
    api.version = CG_STRUCT_VERSION_1;
    api.abi_version = CG_ABI_VERSION;
    api.get_info = get_info;
    api.create_device = create_device;
    api.destroy_device = destroy_device;
    api.resize_device = resize_device;
    api.capture_screenshot = capture_screenshot;

#if defined(CG_TEST_LEGACY_TABLE)
    api.size = (cg_uint32)offsetof(cg_plugin_api, create_scene);
#endif
    const cg_uint32 capacity = in_out_api->size;
    const cg_uint32 tableSize = api.size;
    const cg_uint32 written = capacity < tableSize ? capacity : tableSize;
    api.size = written;
    memcpy(in_out_api, &api, written);
    return CG_SUCCESS;
}
