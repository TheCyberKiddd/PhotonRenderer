#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/software/depth_buffer.h>
#include <cyber/software/framebuffer.h>
#include <cyber/software/pipeline.h>
#include <cyber/software/present_target.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cybergraphics/api.h>
#include <limits>
#include <mutex>
#include <new>

struct cg_device_t
{
    cyber::memory::HeapAllocator allocator;
    cyber::software::Framebuffer color;
    cyber::software::DepthBuffer depth;
    struct cg_scene_t* scenes = nullptr;
    struct cg_mesh_t* meshes = nullptr;
    struct cg_material_t* materials = nullptr;
    struct cg_device_t* nextDevice = nullptr;
    cg_uint32 width = 0;
    cg_uint32 height = 0;
    cg_error_callback errorCallback = nullptr;
    void* errorUserData = nullptr;
};

struct cg_scene_t
{
    cg_device owner = nullptr;
    struct cg_scene_t* next = nullptr;
    struct cg_view_t* views = nullptr;
    struct cg_instance_t* instances = nullptr;
};

struct cg_view_t
{
    cg_scene owner = nullptr;
    struct cg_view_t* next = nullptr;
    cg_view_desc desc{};
};

struct cg_mesh_t
{
    cg_device owner = nullptr;
    struct cg_mesh_t* next = nullptr;
    cg_vertex* vertices = nullptr;
    cg_uint32 vertexCount = 0;
    cg_uint32* indices = nullptr;
    cg_uint32 indexCount = 0;
};

struct cg_material_t
{
    cg_device owner = nullptr;
    struct cg_material_t* next = nullptr;
    cg_material_desc desc{};
};

struct cg_instance_t
{
    cg_scene owner = nullptr;
    struct cg_instance_t* next = nullptr;
    cg_instance_desc desc{};
};

namespace
{

using cyber::core::i32;
using cyber::core::usize;
using cyber::math::Mat4;

std::mutex g_pluginMutex;
cg_device g_devices = nullptr;

template <typename T>
T* AllocateObject() noexcept
{
    void* storage = std::calloc(1, sizeof(T));
    return storage != nullptr ? new (storage) T{} : nullptr;
}

template <typename T>
void ReleaseObject(T* object) noexcept
{
    if (object != nullptr)
    {
        object->~T();
        std::free(object);
    }
}

bool FiniteValues(const float* values, usize count) noexcept
{
    for (usize index = 0; index < count; ++index)
    {
        if (!std::isfinite(values[index]))
        {
            return false;
        }
    }
    return true;
}

bool ContainsDevice(cg_device device) noexcept
{
    for (cg_device current = g_devices; current != nullptr; current = current->nextDevice)
    {
        if (current == device)
        {
            return true;
        }
    }
    return false;
}

cg_scene FindScene(cg_scene scene) noexcept
{
    for (cg_device device = g_devices; device != nullptr; device = device->nextDevice)
    {
        for (cg_scene current = device->scenes; current != nullptr; current = current->next)
        {
            if (current == scene)
            {
                return current;
            }
        }
    }
    return nullptr;
}

bool ContainsView(cg_scene scene, cg_view view) noexcept
{
    for (cg_view current = scene->views; current != nullptr; current = current->next)
    {
        if (current == view)
        {
            return true;
        }
    }
    return false;
}

bool ContainsInstance(cg_scene scene, cg_instance instance) noexcept
{
    for (cg_instance current = scene->instances; current != nullptr; current = current->next)
    {
        if (current == instance)
        {
            return true;
        }
    }
    return false;
}

bool StructVersion(const void* value, cg_uint32 size, cg_uint32 version) noexcept
{
    if (value == nullptr || size < sizeof(cg_uint32) * 2)
    {
        return false;
    }
    const auto* header = static_cast<const cg_uint32*>(value);
    return header[0] >= size && header[1] == version;
}

Mat4 ReadMatrix(const float value[16]) noexcept
{
    Mat4 matrix = Mat4::Zero();
    for (usize column = 0; column < 4; ++column)
    {
        for (usize row = 0; row < 4; ++row)
        {
            matrix[column][row] = value[column * 4 + row];
        }
    }
    return matrix;
}

bool ContainsScene(cg_device device, cg_scene scene) noexcept
{
    for (cg_scene current = device->scenes; current != nullptr; current = current->next)
    {
        if (current == scene)
        {
            return true;
        }
    }
    return false;
}

bool ContainsMesh(cg_device device, cg_mesh mesh) noexcept
{
    for (cg_mesh current = device->meshes; current != nullptr; current = current->next)
    {
        if (current == mesh)
        {
            return true;
        }
    }
    return false;
}

bool ContainsMaterial(cg_device device, cg_material material) noexcept
{
    for (cg_material current = device->materials; current != nullptr; current = current->next)
    {
        if (current == material)
        {
            return true;
        }
    }
    return false;
}

cg_result GetInfo(cg_plugin_info* output) noexcept
{
    if (!StructVersion(output, static_cast<cg_uint32>(sizeof(cg_plugin_info)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    output->name = "CyberGraphics Software Renderer";
    output->capabilities =
        CG_CAPABILITY_SCENE_RENDERING | CG_CAPABILITY_SCREENSHOT | CG_CAPABILITY_RESIZE;
    return CG_SUCCESS;
}

cg_result CreateDevice(const cg_device_desc* desc, cg_device* output) noexcept
{
    if (output == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    if (!StructVersion(desc, static_cast<cg_uint32>(offsetof(cg_device_desc, native_window)),
                       CG_STRUCT_VERSION_1) ||
        desc->width == 0 || desc->height == 0 || desc->width > 16384 || desc->height > 16384)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }

    void* storage = std::calloc(1, sizeof(cg_device_t));
    if (storage == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    cg_device device = new (storage) cg_device_t{};
    cyber::software::FramebufferDesc colorDesc{};
    colorDesc.width = static_cast<i32>(desc->width);
    colorDesc.height = static_cast<i32>(desc->height);
    colorDesc.format = cyber::software::PixelFormat::RGBA32F;
    auto color = cyber::software::Framebuffer::Create(colorDesc, device->allocator);
    if (!color)
    {
        device->~cg_device_t();
        std::free(storage);
        return CG_ERROR_OUT_OF_MEMORY;
    }
    cyber::software::DepthBufferDesc depthDesc{};
    depthDesc.width = colorDesc.width;
    depthDesc.height = colorDesc.height;
    auto depth = cyber::software::DepthBuffer::Create(depthDesc, device->allocator);
    if (!depth)
    {
        device->~cg_device_t();
        std::free(storage);
        return CG_ERROR_OUT_OF_MEMORY;
    }
    device->color = std::move(color.Value());
    device->depth = std::move(depth.Value());
    device->width = desc->width;
    device->height = desc->height;
    device->errorCallback = desc->error_callback;
    device->errorUserData = desc->error_user_data;
    {
        std::lock_guard lock(g_pluginMutex);
        device->nextDevice = g_devices;
        g_devices = device;
    }
    *output = device;
    return CG_SUCCESS;
}

cg_result DestroyDevice(cg_device device) noexcept
{
    if (device == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (device->scenes != nullptr || device->meshes != nullptr || device->materials != nullptr)
    {
        return CG_ERROR_INVALID_STATE;
    }
    cg_device* deviceLink = &g_devices;
    while (*deviceLink != device)
    {
        deviceLink = &(*deviceLink)->nextDevice;
    }
    *deviceLink = device->nextDevice;
    device->~cg_device_t();
    std::free(device);
    return CG_SUCCESS;
}

cg_result ResizeDevice(cg_device device, cg_uint32 width, cg_uint32 height) noexcept
{
    if (device == nullptr || width == 0 || height == 0 || width > 16384 || height > 16384)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cyber::software::FramebufferDesc colorDesc{};
    colorDesc.width = static_cast<i32>(width);
    colorDesc.height = static_cast<i32>(height);
    colorDesc.format = cyber::software::PixelFormat::RGBA32F;
    auto color = cyber::software::Framebuffer::Create(colorDesc, device->allocator);
    if (!color)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    cyber::software::DepthBufferDesc depthDesc{};
    depthDesc.width = colorDesc.width;
    depthDesc.height = colorDesc.height;
    auto depth = cyber::software::DepthBuffer::Create(depthDesc, device->allocator);
    if (!depth)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    device->color = std::move(color.Value());
    device->depth = std::move(depth.Value());
    device->width = width;
    device->height = height;
    return CG_SUCCESS;
}

cg_result CreateScene(cg_device device, const cg_scene_desc* desc, cg_scene* output) noexcept
{
    if (device == nullptr || output == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_scene_desc)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    auto* scene = AllocateObject<cg_scene_t>();
    if (scene == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        ReleaseObject(scene);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    scene->owner = device;
    scene->next = device->scenes;
    device->scenes = scene;
    *output = scene;
    return CG_SUCCESS;
}

cg_result DestroyScene(cg_device device, cg_scene scene) noexcept
{
    if (device == nullptr || scene == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device) || !ContainsScene(device, scene) || scene->views != nullptr ||
        scene->instances != nullptr)
    {
        return CG_ERROR_INVALID_STATE;
    }
    cg_scene* link = &device->scenes;
    while (*link != scene)
    {
        link = &(*link)->next;
    }
    *link = scene->next;
    ReleaseObject(scene);
    return CG_SUCCESS;
}

cg_result CreateView(cg_scene scene, const cg_view_desc* desc, cg_view* output) noexcept
{
    if (scene == nullptr || output == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_view_desc)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    auto* view = AllocateObject<cg_view_t>();
    if (view == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    std::lock_guard lock(g_pluginMutex);
    if (FindScene(scene) == nullptr)
    {
        ReleaseObject(view);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (!FiniteValues(desc->view, 16) || !FiniteValues(desc->projection, 16) ||
        !FiniteValues(desc->viewport, 4) || !FiniteValues(desc->clear_color, 4))
    {
        ReleaseObject(view);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    view->owner = scene;
    view->desc = *desc;
    view->next = scene->views;
    scene->views = view;
    *output = view;
    return CG_SUCCESS;
}

cg_result UpdateView(cg_view view, const cg_view_desc* desc) noexcept
{
    if (view == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_view_desc)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (!FiniteValues(desc->view, 16) || !FiniteValues(desc->projection, 16) ||
        !FiniteValues(desc->viewport, 4) || !FiniteValues(desc->clear_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    bool found = false;
    for (cg_device device = g_devices; device != nullptr && !found; device = device->nextDevice)
    {
        for (cg_scene scene = device->scenes; scene != nullptr && !found; scene = scene->next)
        {
            found = ContainsView(scene, view);
        }
    }
    if (!found)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    view->desc = *desc;
    return CG_SUCCESS;
}

cg_result DestroyView(cg_scene scene, cg_view view) noexcept
{
    if (scene == nullptr || view == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr || !ContainsView(owner, view))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cg_view* link = &owner->views;
    while (*link != nullptr && *link != view)
    {
        link = &(*link)->next;
    }
    if (*link == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *link = view->next;
    ReleaseObject(view);
    return CG_SUCCESS;
}

cg_result CreateMesh(cg_device device, const cg_mesh_desc* desc, cg_mesh* output) noexcept
{
    if (device == nullptr || output == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_mesh_desc)), CG_STRUCT_VERSION_1) ||
        desc->vertices == nullptr || desc->indices == nullptr || desc->vertex_count < 3 ||
        desc->index_count < 3 || desc->index_count % 3 != 0 ||
        desc->vertex_count > std::numeric_limits<usize>::max() / sizeof(cg_vertex) ||
        desc->index_count > std::numeric_limits<usize>::max() / sizeof(cg_uint32))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    for (cg_uint32 i = 0; i < desc->vertex_count; ++i)
    {
        if (!StructVersion(desc->vertices + i, static_cast<cg_uint32>(sizeof(cg_vertex)),
                           CG_STRUCT_VERSION_1) ||
            !FiniteValues(desc->vertices[i].position, 3) ||
            !FiniteValues(desc->vertices[i].color, 4))
        {
            return CG_ERROR_INVALID_ARGUMENT;
        }
    }
    for (cg_uint32 i = 0; i < desc->index_count; ++i)
    {
        if (desc->indices[i] >= desc->vertex_count)
        {
            return CG_ERROR_OUT_OF_RANGE;
        }
    }

    auto* mesh = AllocateObject<cg_mesh_t>();
    if (mesh == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    mesh->vertices = static_cast<cg_vertex*>(std::malloc(sizeof(cg_vertex) * desc->vertex_count));
    mesh->indices = static_cast<cg_uint32*>(std::malloc(sizeof(cg_uint32) * desc->index_count));
    if (mesh->vertices == nullptr || mesh->indices == nullptr)
    {
        std::free(mesh->vertices);
        std::free(mesh->indices);
        ReleaseObject(mesh);
        return CG_ERROR_OUT_OF_MEMORY;
    }
    std::memcpy(mesh->vertices, desc->vertices, sizeof(cg_vertex) * desc->vertex_count);
    std::memcpy(mesh->indices, desc->indices, sizeof(cg_uint32) * desc->index_count);
    mesh->vertexCount = desc->vertex_count;
    mesh->indexCount = desc->index_count;

    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        std::free(mesh->vertices);
        std::free(mesh->indices);
        ReleaseObject(mesh);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    mesh->owner = device;
    mesh->next = device->meshes;
    device->meshes = mesh;
    *output = mesh;
    return CG_SUCCESS;
}

cg_result DestroyMesh(cg_device device, cg_mesh mesh) noexcept
{
    if (device == nullptr || mesh == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device) || !ContainsMesh(device, mesh))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    for (cg_scene scene = device->scenes; scene != nullptr; scene = scene->next)
    {
        for (cg_instance instance = scene->instances; instance != nullptr;
             instance = instance->next)
        {
            if (instance->desc.mesh == mesh)
            {
                return CG_ERROR_INVALID_STATE;
            }
        }
    }
    cg_mesh* link = &device->meshes;
    while (*link != mesh)
    {
        link = &(*link)->next;
    }
    *link = mesh->next;
    std::free(mesh->vertices);
    std::free(mesh->indices);
    ReleaseObject(mesh);
    return CG_SUCCESS;
}

cg_result CreateMaterial(cg_device device, const cg_material_desc* desc,
                         cg_material* output) noexcept
{
    if (device == nullptr || output == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_material_desc)),
                       CG_STRUCT_VERSION_1) ||
        !FiniteValues(desc->base_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    auto* material = AllocateObject<cg_material_t>();
    if (material == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    material->owner = device;
    material->desc = *desc;
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        ReleaseObject(material);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    material->next = device->materials;
    device->materials = material;
    *output = material;
    return CG_SUCCESS;
}

cg_result UpdateMaterial(cg_material material, const cg_material_desc* desc) noexcept
{
    if (material == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_material_desc)),
                       CG_STRUCT_VERSION_1) ||
        !FiniteValues(desc->base_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    bool found = false;
    for (cg_device device = g_devices; device != nullptr && !found; device = device->nextDevice)
    {
        found = ContainsMaterial(device, material);
    }
    if (!found)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    material->desc = *desc;
    return CG_SUCCESS;
}

cg_result DestroyMaterial(cg_device device, cg_material material) noexcept
{
    if (device == nullptr || material == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device) || !ContainsMaterial(device, material))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    for (cg_scene scene = device->scenes; scene != nullptr; scene = scene->next)
    {
        for (cg_instance instance = scene->instances; instance != nullptr;
             instance = instance->next)
        {
            if (instance->desc.material == material)
            {
                return CG_ERROR_INVALID_STATE;
            }
        }
    }
    cg_material* link = &device->materials;
    while (*link != material)
    {
        link = &(*link)->next;
    }
    *link = material->next;
    ReleaseObject(material);
    return CG_SUCCESS;
}

cg_result CreateInstance(cg_scene scene, const cg_instance_desc* desc, cg_instance* output) noexcept
{
    if (scene == nullptr || output == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_instance_desc)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    std::lock_guard lock(g_pluginMutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (!FiniteValues(desc->transform, 16))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cg_device device = owner->owner;
    if (!ContainsScene(device, scene) || !ContainsMesh(device, desc->mesh) ||
        !ContainsMaterial(device, desc->material))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    auto* instance = AllocateObject<cg_instance_t>();
    if (instance == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    instance->owner = scene;
    instance->desc = *desc;
    instance->next = scene->instances;
    scene->instances = instance;
    *output = instance;
    return CG_SUCCESS;
}

cg_result UpdateInstance(cg_instance instance, const cg_instance_desc* desc) noexcept
{
    if (instance == nullptr ||
        !StructVersion(desc, static_cast<cg_uint32>(sizeof(cg_instance_desc)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (!FiniteValues(desc->transform, 16))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    cg_scene owner = nullptr;
    for (cg_device candidate = g_devices; candidate != nullptr && owner == nullptr;
         candidate = candidate->nextDevice)
    {
        for (cg_scene scene = candidate->scenes; scene != nullptr; scene = scene->next)
        {
            if (ContainsInstance(scene, instance))
            {
                owner = scene;
                break;
            }
        }
    }
    if (owner == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cg_device device = owner->owner;
    if (!ContainsMesh(device, desc->mesh) || !ContainsMaterial(device, desc->material))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    instance->desc = *desc;
    return CG_SUCCESS;
}

cg_result DestroyInstance(cg_scene scene, cg_instance instance) noexcept
{
    if (scene == nullptr || instance == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr || !ContainsInstance(owner, instance))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cg_instance* link = &owner->instances;
    while (*link != instance)
    {
        link = &(*link)->next;
    }
    *link = instance->next;
    ReleaseObject(instance);
    return CG_SUCCESS;
}

cg_result SubmitFrame(cg_device device, cg_scene scene, cg_view view) noexcept
{
    if (device == nullptr || scene == nullptr || view == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device) || !ContainsScene(device, scene) || !ContainsView(scene, view) ||
        view->owner != scene)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    const cg_view_desc& viewDesc = view->desc;
    const float* viewport = viewDesc.viewport;
    if (!(viewport[2] > 0.0f) || !(viewport[3] > 0.0f) || viewport[0] < 0.0f ||
        viewport[1] < 0.0f || viewport[0] + viewport[2] > static_cast<float>(device->width) ||
        viewport[1] + viewport[3] > static_cast<float>(device->height))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }

    device->color.Clear(cyber::math::Color{viewDesc.clear_color[0], viewDesc.clear_color[1],
                                           viewDesc.clear_color[2], viewDesc.clear_color[3]});
    device->depth.Clear();
    for (cg_instance instance = scene->instances; instance != nullptr; instance = instance->next)
    {
        cg_mesh mesh = instance->desc.mesh;
        cg_material material = instance->desc.material;
        if (!ContainsMesh(device, mesh) || !ContainsMaterial(device, material))
        {
            return CG_ERROR_INVALID_STATE;
        }
        if (mesh->indexCount > std::numeric_limits<usize>::max() / sizeof(cyber::software::Vertex))
        {
            return CG_ERROR_LIMIT_EXCEEDED;
        }
        auto* vertices = static_cast<cyber::software::Vertex*>(device->allocator.Allocate(
            sizeof(cyber::software::Vertex) * mesh->indexCount, alignof(cyber::software::Vertex)));
        if (vertices == nullptr)
        {
            return CG_ERROR_OUT_OF_MEMORY;
        }
        for (cg_uint32 i = 0; i < mesh->indexCount; ++i)
        {
            const cg_vertex& source = mesh->vertices[mesh->indices[i]];
            vertices[i].position =
                cyber::math::Vec3{source.position[0], source.position[1], source.position[2]};
            vertices[i].color = cyber::math::Color{source.color[0] * material->desc.base_color[0],
                                                   source.color[1] * material->desc.base_color[1],
                                                   source.color[2] * material->desc.base_color[2],
                                                   source.color[3] * material->desc.base_color[3]};
        }

        cyber::software::PipelineDesc pipeline{};
        pipeline.model = ReadMatrix(instance->desc.transform);
        pipeline.view = ReadMatrix(viewDesc.view);
        pipeline.projection = ReadMatrix(viewDesc.projection);
        pipeline.viewport =
            cyber::software::Viewport{viewport[0], viewport[1], viewport[2], viewport[3]};
        pipeline.cullMode = cyber::software::CullMode::None;
        const cyber::core::Status<cyber::core::Error> drawn = cyber::software::DrawTriangles(
            pipeline,
            cyber::memory::Span<const cyber::software::Vertex>(vertices, mesh->indexCount),
            device->color, device->depth, nullptr, device->allocator);
        device->allocator.Deallocate(vertices);
        if (!drawn)
        {
            return CG_ERROR_INTERNAL;
        }
    }
    return CG_SUCCESS;
}

cg_result CaptureScreenshot(cg_device device, cg_image* image) noexcept
{
    if (device == nullptr ||
        !StructVersion(image, static_cast<cg_uint32>(sizeof(cg_image)), CG_STRUCT_VERSION_1))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_pluginMutex);
    if (!ContainsDevice(device))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if ((image->width != 0 && image->width != device->width) ||
        (image->height != 0 && image->height != device->height))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    image->width = device->width;
    image->height = device->height;
    const usize pixelBytes = image->format == CG_PIXEL_FORMAT_RGBA8_SRGB       ? 4
                             : image->format == CG_PIXEL_FORMAT_RGBA32F_LINEAR ? 16
                                                                               : 0;
    if (pixelBytes == 0 || image->width > std::numeric_limits<usize>::max() / pixelBytes)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    const usize rowBytes = static_cast<usize>(image->width) * pixelBytes;
    if constexpr (sizeof(usize) < sizeof(cg_uint64))
    {
        if (image->row_stride_bytes > std::numeric_limits<usize>::max() ||
            image->capacity_bytes > std::numeric_limits<usize>::max())
        {
            return CG_ERROR_LIMIT_EXCEEDED;
        }
    }
    const usize stride =
        image->row_stride_bytes == 0 ? rowBytes : static_cast<usize>(image->row_stride_bytes);
    if (stride < rowBytes ||
        (image->height > 0 && stride > std::numeric_limits<usize>::max() / image->height))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    image->required_bytes = static_cast<cg_uint64>(stride * image->height);
    if (image->pixels == nullptr || image->capacity_bytes < image->required_bytes)
    {
        return CG_ERROR_BUFFER_TOO_SMALL;
    }
    if (image->format == CG_PIXEL_FORMAT_RGBA8_SRGB)
    {
        return cyber::software::PackPresentPixels(device->color,
                                                  cyber::memory::Span<cyber::core::u8>(
                                                      static_cast<cyber::core::u8*>(image->pixels),
                                                      static_cast<usize>(image->capacity_bytes)),
                                                  stride)
                   ? CG_SUCCESS
                   : CG_ERROR_INTERNAL;
    }

    for (cg_uint32 y = 0; y < image->height; ++y)
    {
        const auto* source =
            reinterpret_cast<const cyber::core::u8*>(device->color.Row(static_cast<i32>(y)));
        std::memcpy(static_cast<cyber::core::u8*>(image->pixels) + stride * y, source, rowBytes);
    }
    return CG_SUCCESS;
}

cg_result GetPluginApi(cg_uint32 requestedVersion, cg_plugin_api* api) noexcept
{
    if (api == nullptr || api->size < offsetof(cg_plugin_api, get_info) + sizeof(api->get_info))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (requestedVersion != CG_ABI_VERSION)
    {
        return CG_ERROR_INCOMPATIBLE_VERSION;
    }
    const cg_uint32 capacity = api->size;
    cg_plugin_api table{};
    table.size = static_cast<cg_uint32>(sizeof(table));
    table.version = CG_STRUCT_VERSION_1;
    table.abi_version = CG_ABI_VERSION;
    table.get_info = GetInfo;
    table.create_device = CreateDevice;
    table.destroy_device = DestroyDevice;
    table.resize_device = ResizeDevice;
    table.create_scene = CreateScene;
    table.destroy_scene = DestroyScene;
    table.create_view = CreateView;
    table.update_view = UpdateView;
    table.destroy_view = DestroyView;
    table.create_mesh = CreateMesh;
    table.destroy_mesh = DestroyMesh;
    table.create_material = CreateMaterial;
    table.update_material = UpdateMaterial;
    table.destroy_material = DestroyMaterial;
    table.create_instance = CreateInstance;
    table.update_instance = UpdateInstance;
    table.destroy_instance = DestroyInstance;
    table.submit_frame = SubmitFrame;
    table.capture_screenshot = CaptureScreenshot;
    const cg_uint32 written = capacity < sizeof(table) ? capacity : sizeof(table);
    table.size = written;
    std::memcpy(api, &table, written);
    return CG_SUCCESS;
}

}  // namespace

extern "C" CG_PLUGIN_EXPORT cg_result CG_CALL cgGetPluginApi(cg_uint32 requested_abi_version,
                                                             cg_plugin_api* in_out_api)
{
    return GetPluginApi(requested_abi_version, in_out_api);
}
