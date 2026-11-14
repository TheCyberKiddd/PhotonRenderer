#include <cybergraphics/api.h>

#include <stddef.h>

_Static_assert(sizeof(cg_result) == sizeof(int32_t), "cg_result must remain 32-bit");
_Static_assert(sizeof(cg_uint32) == 4, "cg_uint32 must remain 32-bit");
_Static_assert(sizeof(cg_uint64) == 8, "cg_uint64 must remain 64-bit");
_Static_assert(offsetof(cg_device_desc, size) == 0, "size must be the first field");
_Static_assert(offsetof(cg_device_desc, version) == sizeof(cg_uint32),
               "version must follow size");
_Static_assert(offsetof(cg_plugin_api, size) == 0, "table size must be the first field");
_Static_assert(offsetof(cg_plugin_api, version) == sizeof(cg_uint32),
               "table version must follow size");

int main(void)
{
    cg_plugin_api api = {0};
    cg_device_desc device = {0};

    api.size = (cg_uint32)sizeof(api);
    api.version = CG_STRUCT_VERSION_1;
    device.size = (cg_uint32)sizeof(device);
    device.version = CG_STRUCT_VERSION_1;

    return api.size == sizeof(cg_plugin_api) && device.size == sizeof(cg_device_desc) &&
                   CG_ABI_VERSION == 1U && CG_PLUGIN_ENTRY_POINT[0] == 'c'
               ? 0
               : 1;
}
