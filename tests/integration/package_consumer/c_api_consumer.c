#include <cybergraphics/api.h>

#include <stddef.h>

int cybergraphics_c_api_consumer(void)
{
    cg_plugin_api api = {0};
    cg_device_desc device = {0};

    api.size = (cg_uint32)sizeof(api);
    api.version = CG_STRUCT_VERSION_1;
    device.size = (cg_uint32)sizeof(device);
    device.version = CG_STRUCT_VERSION_1;

    return offsetof(cg_plugin_api, size) == 0 &&
                   offsetof(cg_device_desc, size) == 0 &&
                   api.size == sizeof(cg_plugin_api) &&
                   device.size == sizeof(cg_device_desc) && CG_ABI_VERSION == 1U
               ? 0
               : 1;
}
