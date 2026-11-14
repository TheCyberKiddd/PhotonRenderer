#ifndef CYBERGRAPHICS_API_H
#define CYBERGRAPHICS_API_H

#include <stdint.h>

#if defined(_WIN32)
#define CG_CALL __cdecl
#if defined(CG_PLUGIN_BUILD)
#define CG_PLUGIN_EXPORT __declspec(dllexport)
#else
#define CG_PLUGIN_EXPORT
#endif
#else
#define CG_CALL
#if defined(CG_PLUGIN_BUILD) && (defined(__GNUC__) || defined(__clang__))
#define CG_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define CG_PLUGIN_EXPORT
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef int32_t cg_result;
typedef uint32_t cg_uint32;
typedef uint64_t cg_uint64;

#define CG_SUCCESS ((cg_result)0)
#define CG_ERROR_INVALID_ARGUMENT ((cg_result)1)
#define CG_ERROR_INCOMPATIBLE_VERSION ((cg_result)2)
#define CG_ERROR_UNSUPPORTED ((cg_result)3)
#define CG_ERROR_OUT_OF_MEMORY ((cg_result)4)
#define CG_ERROR_BUFFER_TOO_SMALL ((cg_result)5)
#define CG_ERROR_INVALID_STATE ((cg_result)6)
#define CG_ERROR_NOT_FOUND ((cg_result)7)
#define CG_ERROR_LIMIT_EXCEEDED ((cg_result)8)
#define CG_ERROR_INTERNAL ((cg_result)9)
#define CG_ERROR_OUT_OF_RANGE ((cg_result)10)

#define CG_ABI_VERSION 1U
#define CG_STRUCT_VERSION_1 1U
#define CG_PLUGIN_ENTRY_POINT "cgGetPluginApi"

#define CG_CAPABILITY_SCENE_RENDERING (UINT64_C(1) << 0)
#define CG_CAPABILITY_DEBUG_LINES (UINT64_C(1) << 1)
#define CG_CAPABILITY_SCREENSHOT (UINT64_C(1) << 2)
#define CG_CAPABILITY_RESIZE (UINT64_C(1) << 3)
#define CG_CAPABILITY_LIGHTING (UINT64_C(1) << 4)

typedef struct cg_device_t* cg_device;
typedef struct cg_scene_t* cg_scene;
typedef struct cg_view_t* cg_view;
typedef struct cg_mesh_t* cg_mesh;
typedef struct cg_material_t* cg_material;
typedef struct cg_instance_t* cg_instance;
typedef struct cg_light_t* cg_light;

typedef void(CG_CALL* cg_error_callback)(void* user_data, cg_result result, const char* message);

typedef struct cg_device_desc
{
    cg_uint32 size;
    cg_uint32 version;
    cg_uint32 width;
    cg_uint32 height;
    cg_error_callback error_callback;
    void* error_user_data;
    // Optional borrowed native window required by presentation-capable renderers.
    // Windows plugins interpret this as HWND; null is valid for headless plugins.
    const void* native_window;
} cg_device_desc;

typedef struct cg_scene_desc
{
    cg_uint32 size;
    cg_uint32 version;
} cg_scene_desc;

typedef struct cg_view_desc
{
    cg_uint32 size;
    cg_uint32 version;
    float view[16];
    float projection[16];
    float viewport[4];
    float clear_color[4];
} cg_view_desc;

typedef struct cg_vertex
{
    cg_uint32 size;
    cg_uint32 version;
    float position[3];
    float color[4];
} cg_vertex;

typedef struct cg_mesh_desc
{
    cg_uint32 size;
    cg_uint32 version;
    const cg_vertex* vertices;
    cg_uint32 vertex_count;
    const cg_uint32* indices;
    cg_uint32 index_count;
} cg_mesh_desc;

typedef struct cg_material_desc
{
    cg_uint32 size;
    cg_uint32 version;
    float base_color[4];
} cg_material_desc;

typedef struct cg_instance_desc
{
    cg_uint32 size;
    cg_uint32 version;
    cg_mesh mesh;
    cg_material material;
    float transform[16];
} cg_instance_desc;

#define CG_LIGHT_DIRECTIONAL 0U
#define CG_LIGHT_POINT 1U
#define CG_LIGHT_SPOT 2U

typedef struct cg_light_desc
{
    cg_uint32 size;
    cg_uint32 version;
    cg_uint32 type;
    float position[3];
    float direction[3];
    float color[3];
    float intensity;
    float range;
    float inner_cone_radians;
    float outer_cone_radians;
} cg_light_desc;

typedef struct cg_debug_line
{
    cg_uint32 size;
    cg_uint32 version;
    float start[3];
    float end[3];
    float color[4];
} cg_debug_line;

#define CG_PIXEL_FORMAT_RGBA8_SRGB 1U
#define CG_PIXEL_FORMAT_RGBA32F_LINEAR 2U

typedef struct cg_image
{
    cg_uint32 size;
    cg_uint32 version;
    cg_uint32 format;
    cg_uint32 width;
    cg_uint32 height;
    cg_uint64 row_stride_bytes;
    cg_uint64 capacity_bytes;
    cg_uint64 required_bytes;
    void* pixels;
} cg_image;

typedef struct cg_plugin_info
{
    cg_uint32 size;
    cg_uint32 version;
    const char* name;
    cg_uint64 capabilities;
} cg_plugin_info;

typedef struct cg_plugin_api
{
    cg_uint32 size;
    cg_uint32 version;
    cg_uint32 abi_version;
    cg_uint32 reserved;
    cg_result(CG_CALL* get_info)(cg_plugin_info* out_info);
    cg_result(CG_CALL* create_device)(const cg_device_desc* desc, cg_device* out_device);
    cg_result(CG_CALL* destroy_device)(cg_device device);
    cg_result(CG_CALL* resize_device)(cg_device device, cg_uint32 width, cg_uint32 height);
    cg_result(CG_CALL* create_scene)(cg_device device, const cg_scene_desc* desc,
                                     cg_scene* out_scene);
    cg_result(CG_CALL* destroy_scene)(cg_device device, cg_scene scene);
    cg_result(CG_CALL* create_view)(cg_scene scene, const cg_view_desc* desc, cg_view* out_view);
    cg_result(CG_CALL* update_view)(cg_view view, const cg_view_desc* desc);
    cg_result(CG_CALL* destroy_view)(cg_scene scene, cg_view view);
    cg_result(CG_CALL* create_mesh)(cg_device device, const cg_mesh_desc* desc, cg_mesh* out_mesh);
    cg_result(CG_CALL* destroy_mesh)(cg_device device, cg_mesh mesh);
    cg_result(CG_CALL* create_material)(cg_device device, const cg_material_desc* desc,
                                        cg_material* out_material);
    cg_result(CG_CALL* update_material)(cg_material material, const cg_material_desc* desc);
    cg_result(CG_CALL* destroy_material)(cg_device device, cg_material material);
    cg_result(CG_CALL* create_instance)(cg_scene scene, const cg_instance_desc* desc,
                                        cg_instance* out_instance);
    cg_result(CG_CALL* update_instance)(cg_instance instance, const cg_instance_desc* desc);
    cg_result(CG_CALL* destroy_instance)(cg_scene scene, cg_instance instance);
    cg_result(CG_CALL* create_light)(cg_scene scene, const cg_light_desc* desc,
                                     cg_light* out_light);
    cg_result(CG_CALL* update_light)(cg_light light, const cg_light_desc* desc);
    cg_result(CG_CALL* destroy_light)(cg_scene scene, cg_light light);
    cg_result(CG_CALL* draw_debug_lines)(cg_scene scene, const cg_debug_line* lines,
                                         cg_uint32 line_count);
    cg_result(CG_CALL* submit_frame)(cg_device device, cg_scene scene, cg_view view);
    cg_result(CG_CALL* capture_screenshot)(cg_device device, cg_image* in_out_image);
} cg_plugin_api;

typedef cg_result(CG_CALL* cg_get_plugin_api_fn)(cg_uint32 requested_abi_version,
                                                 cg_plugin_api* in_out_api);

CG_PLUGIN_EXPORT cg_result CG_CALL cgGetPluginApi(cg_uint32 requested_abi_version,
                                                  cg_plugin_api* in_out_api);

#ifdef __cplusplus
}
#endif

#endif
