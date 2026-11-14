#include <cyber/core/types.h>
#include <cyber/math/mat4.h>
#include <cyber/memory/heap_allocator.h>
#include <cyber/memory/span.h>
#include <cyber/rhi/backend.h>
#include <cyber/rhi/binding.h>
#include <cyber/rhi/buffer.h>
#include <cyber/rhi/command_list.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/device.h>
#include <cyber/rhi/pipeline.h>
#include <cyber/rhi/render_pass.h>
#include <cyber/rhi/resource_state.h>
#include <cyber/rhi/shader.h>
#include <cyber/rhi/submission.h>
#include <cyber/rhi/swapchain.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cybergraphics/api.h>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

struct cg_device_t
{
    cyber::memory::HeapAllocator allocator;
    cyber::rhi::Device device;
    cyber::rhi::SwapChain swapchain;
    cyber::rhi::PipelineHandle pipeline{};
    cyber::rhi::PipelineLayoutHandle pipelineLayout{};
    cyber::rhi::BindGroupLayoutHandle drawLayout{};
    cyber::rhi::ShaderHandle vertexShader{};
    cyber::rhi::ShaderHandle pixelShader{};
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
    cyber::rhi::BufferHandle vertices{};
    cyber::rhi::BufferHandle indices{};
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
    cyber::rhi::BufferHandle constants{};
    cyber::rhi::BufferHandle staging{};
    cyber::rhi::BindGroupHandle bindGroup{};
    bool stagingReady = false;
};

namespace
{

using cyber::core::i32;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::usize;
using cyber::math::Mat4;
using cyber::memory::Byte;
using cyber::memory::Span;
using cyber::rhi::BindGroupHandle;
using cyber::rhi::BufferBarrier;
using cyber::rhi::BufferCopy;
using cyber::rhi::BufferHandle;
using cyber::rhi::BufferUsage;
using cyber::rhi::CommandList;
using cyber::rhi::CommandListHandle;
using cyber::rhi::Device;
using cyber::rhi::DeviceDesc;
using cyber::rhi::Format;
using cyber::rhi::PipelineHandle;
using cyber::rhi::ResourceState;
using cyber::rhi::ShaderHandle;
using cyber::rhi::TextureBarrier;
using cyber::rhi::TextureHandle;
using cyber::rhi::TextureViewHandle;

std::recursive_mutex g_mutex;
cg_device g_devices = nullptr;

constexpr char kVertexShader[] =
    "cbuffer DrawConstants : register(b0, space3) {\n"
    "  float4x4 objectToClip : packoffset(c0);\n"
    "  float4 baseColor : packoffset(c4);\n"
    "};\n"
    "struct Input { float3 position : TEXCOORD0; float4 color : TEXCOORD1; };\n"
    "struct Output { float4 position : SV_Position; float4 color : TEXCOORD0; };\n"
    "Output main(Input input) {\n"
    "  Output output;\n"
    "  output.position = mul(objectToClip, float4(input.position, 1.0));\n"
    "  output.position.y = -output.position.y;\n"
    "  output.color = input.color;\n"
    "  return output;\n"
    "}\n";
constexpr char kPixelShader[] =
    "cbuffer DrawConstants : register(b0, space3) {\n"
    "  float4x4 objectToClip : packoffset(c0);\n"
    "  float4 baseColor : packoffset(c4);\n"
    "};\n"
    "float4 main(float4 position : SV_Position, float4 color : TEXCOORD0) : SV_Target {\n"
    "  return color * baseColor;\n"
    "}\n";

struct Vertex
{
    float position[3];
    float color[4];
};
static_assert(sizeof(Vertex) == sizeof(float) * 7);

struct DrawConstants
{
    float objectToClip[16];
    float baseColor[4];
};
static_assert(sizeof(DrawConstants) == 80);

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

bool StructVersion(const void* value, usize size) noexcept
{
    if (value == nullptr)
    {
        return false;
    }
    const auto* header = static_cast<const cg_uint32*>(value);
    return header[0] >= size && header[1] == CG_STRUCT_VERSION_1;
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

bool ContainsDevice(cg_device device) noexcept;

cg_result NotifyCallback(cg_error_callback callback, void* userData, cg_result result,
                         const char* message) noexcept
{
    if (callback != nullptr && result != CG_SUCCESS)
    {
        callback(userData, result, message);
    }
    return result;
}

cg_result Notify(cg_device device, cg_result result, const char* message) noexcept
{
    if (device != nullptr && result != CG_SUCCESS)
    {
        std::lock_guard lock(g_mutex);
        if (ContainsDevice(device))
        {
            return NotifyCallback(device->errorCallback, device->errorUserData, result, message);
        }
    }
    return result;
}

cg_result MapError(cyber::core::ErrorCode code) noexcept
{
    using cyber::core::ErrorCode;
    switch (code)
    {
    case ErrorCode::Ok:
        return CG_SUCCESS;
    case ErrorCode::InvalidArgument:
        return CG_ERROR_INVALID_ARGUMENT;
    case ErrorCode::OutOfRange:
        return CG_ERROR_OUT_OF_RANGE;
    case ErrorCode::NotFound:
        return CG_ERROR_NOT_FOUND;
    case ErrorCode::AlreadyExists:
        return CG_ERROR_INVALID_STATE;
    case ErrorCode::Unsupported:
        return CG_ERROR_UNSUPPORTED;
    case ErrorCode::Overflow:
        return CG_ERROR_LIMIT_EXCEEDED;
    case ErrorCode::DeviceLost:
    case ErrorCode::Busy:
    case ErrorCode::ParseError:
    case ErrorCode::Internal:
        return CG_ERROR_INTERNAL;
    }
    return CG_ERROR_INTERNAL;
}

cg_result Notify(cg_device device, const cyber::core::Error& error, const char* operation) noexcept
{
    return Notify(device, MapError(error.code),
                  error.message != nullptr ? error.message : operation);
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

cg_mesh FindMesh(cg_device device, cg_mesh mesh) noexcept
{
    for (cg_mesh current = device->meshes; current != nullptr; current = current->next)
    {
        if (current == mesh)
        {
            return current;
        }
    }
    return nullptr;
}

cg_material FindMaterial(cg_device device, cg_material material) noexcept
{
    for (cg_material current = device->materials; current != nullptr; current = current->next)
    {
        if (current == material)
        {
            return current;
        }
    }
    return nullptr;
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

bool CreatePipeline(cg_device device) noexcept
{
    cyber::rhi::BindGroupLayoutEntry drawEntry{};
    drawEntry.binding = 0;
    drawEntry.type = cyber::rhi::BindingType::UniformBuffer;
    drawEntry.visibility = cyber::rhi::ShaderStageBit(cyber::rhi::ShaderStage::Vertex) |
                           cyber::rhi::ShaderStageBit(cyber::rhi::ShaderStage::Pixel);
    drawEntry.shaderRegister = 0;
    cyber::rhi::BindGroupLayoutDesc drawLayoutDesc{};
    drawLayoutDesc.entries = Span<const cyber::rhi::BindGroupLayoutEntry>(&drawEntry, 1);
    auto drawLayout = device->device.CreateBindGroupLayout(drawLayoutDesc);
    if (!drawLayout)
    {
        Notify(device, drawLayout.Error(), "CreateBindGroupLayout");
        return false;
    }
    device->drawLayout = drawLayout.Value();

    cyber::rhi::PipelineLayoutDesc layoutDesc{};
    layoutDesc.groups[static_cast<u32>(cyber::rhi::BindGroupSlot::PerDraw)] = device->drawLayout;
    auto layout = device->device.CreatePipelineLayout(layoutDesc);
    if (!layout)
    {
        Notify(device, layout.Error(), "CreatePipelineLayout");
        return false;
    }
    device->pipelineLayout = layout.Value();

    cyber::rhi::ShaderDesc vertexDesc{};
    vertexDesc.stage = cyber::rhi::ShaderStage::Vertex;
    vertexDesc.format = cyber::rhi::ShaderBytecodeFormat::HlslSource;
    vertexDesc.code =
        Span<const Byte>(reinterpret_cast<const Byte*>(kVertexShader), sizeof(kVertexShader) - 1);
    vertexDesc.entryPoint = "main";
    vertexDesc.label = "CyberGraphics renderer vertex shader";
    auto vertex = device->device.CreateShader(vertexDesc);
    if (!vertex)
    {
        Notify(device, vertex.Error(), "CreateShader vertex");
        return false;
    }
    device->vertexShader = vertex.Value();

    cyber::rhi::ShaderDesc pixelDesc{};
    pixelDesc.stage = cyber::rhi::ShaderStage::Pixel;
    pixelDesc.format = cyber::rhi::ShaderBytecodeFormat::HlslSource;
    pixelDesc.code =
        Span<const Byte>(reinterpret_cast<const Byte*>(kPixelShader), sizeof(kPixelShader) - 1);
    pixelDesc.entryPoint = "main";
    pixelDesc.label = "CyberGraphics renderer pixel shader";
    auto pixel = device->device.CreateShader(pixelDesc);
    if (!pixel)
    {
        Notify(device, pixel.Error(), "CreateShader pixel");
        return false;
    }
    device->pixelShader = pixel.Value();

    cyber::rhi::VertexBinding binding{};
    binding.strideBytes = sizeof(Vertex);
    cyber::rhi::VertexAttribute attributes[] = {
        {0, 0, Format::R32G32B32Float, static_cast<u32>(offsetof(Vertex, position))},
        {1, 0, Format::RGBA32Float, static_cast<u32>(offsetof(Vertex, color))},
    };
    cyber::rhi::ColorAttachmentState attachment{};
    attachment.format = Format::BGRA8Unorm;
    cyber::rhi::GraphicsPipelineDesc desc{};
    desc.layout = device->pipelineLayout;
    desc.vertexShader = device->vertexShader;
    desc.pixelShader = device->pixelShader;
    desc.vertexLayout.bindings = Span<const cyber::rhi::VertexBinding>(&binding, 1);
    desc.vertexLayout.attributes = Span<const cyber::rhi::VertexAttribute>(attributes, 2);
    desc.rasterizer.cullMode = cyber::rhi::CullMode::None;
    desc.depthStencil.depthTestEnabled = false;
    desc.depthStencil.depthWriteEnabled = false;
    desc.depthStencilFormat = Format::Count;
    desc.colorAttachments = Span<const cyber::rhi::ColorAttachmentState>(&attachment, 1);
    desc.label = "CyberGraphics renderer scene pipeline";
    auto pipeline = device->device.CreateGraphicsPipeline(desc);
    if (!pipeline)
    {
        Notify(device, pipeline.Error(), "CreateGraphicsPipeline");
        return false;
    }
    device->pipeline = pipeline.Value();
    return true;
}

bool UploadBuffer(cg_device device, BufferUsage usage, const void* data, usize size,
                  BufferHandle& output) noexcept
{
    cyber::rhi::BufferDesc targetDesc{};
    targetDesc.usage = usage | BufferUsage::CopyDestination;
    targetDesc.sizeBytes = size;
    auto target = device->device.CreateBuffer(targetDesc);
    if (!target)
    {
        Notify(device, target.Error(), "CreateBuffer destination");
        return false;
    }
    output = target.Value();

    cyber::rhi::BufferDesc stagingDesc{};
    stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    stagingDesc.sizeBytes = size;
    auto staging = device->device.CreateBuffer(stagingDesc);
    if (!staging)
    {
        Notify(device, staging.Error(), "CreateBuffer staging");
        (void)device->device.DestroyBuffer(output);
        output = {};
        return false;
    }
    const auto bytes = Span<const Byte>(static_cast<const Byte*>(data), size);
    auto written = device->device.WriteBuffer(staging.Value(), 0, bytes);
    if (!written)
    {
        Notify(device, written.Error(), "WriteBuffer staging");
        (void)device->device.DestroyBuffer(staging.Value());
        (void)device->device.DestroyBuffer(output);
        output = {};
        return false;
    }
    auto created = device->device.CreateCommandList(cyber::rhi::CommandListDesc{});
    if (!created)
    {
        Notify(device, created.Error(), "CreateCommandList upload");
        (void)device->device.DestroyBuffer(staging.Value());
        (void)device->device.DestroyBuffer(output);
        output = {};
        return false;
    }
    CommandList list = std::move(created.Value());
    const BufferBarrier barriers[] = {
        {staging.Value(), ResourceState::Undefined, ResourceState::CopySource},
        {output, ResourceState::Undefined, ResourceState::CopyDestination},
    };
    bool ok = list.Begin().IsOk() && list.Barrier(Span<const BufferBarrier>(barriers, 2)).IsOk() &&
              list.CopyBuffer(BufferCopy{staging.Value(), 0, output, 0, size}).IsOk();
    const ResourceState finalState = usage == BufferUsage::Vertex  ? ResourceState::VertexBuffer
                                     : usage == BufferUsage::Index ? ResourceState::IndexBuffer
                                                                   : ResourceState::ConstantBuffer;
    const BufferBarrier ready{output, ResourceState::CopyDestination, finalState};
    ok = ok && list.Barrier(Span<const BufferBarrier>(&ready, 1)).IsOk() && list.End().IsOk();
    if (ok)
    {
        const CommandListHandle handle = list.Handle();
        cyber::rhi::SubmitDesc submit{};
        submit.queue = cyber::rhi::QueueKind::Graphics;
        submit.commandLists = Span<const CommandListHandle>(&handle, 1);
        auto submitted = device->device.Submit(submit);
        if (!submitted)
        {
            Notify(device, submitted.Error(), "Submit upload");
            ok = false;
        }
        else
        {
            auto idle = device->device.WaitIdle();
            if (!idle)
            {
                Notify(device, idle.Error(), "WaitIdle upload");
                ok = false;
            }
        }
    }
    if (!device->device.DestroyBuffer(staging.Value()))
    {
        Notify(device, CG_ERROR_INTERNAL, "DestroyBuffer staging");
        ok = false;
    }
    if (!ok)
    {
        (void)device->device.DestroyBuffer(output);
        output = {};
        return false;
    }
    return true;
}

cg_result GetInfo(cg_plugin_info* output) noexcept
{
    if (!StructVersion(output, sizeof(*output)))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    output->name = "CyberGraphics D3D12 Renderer";
    output->capabilities = CG_CAPABILITY_SCENE_RENDERING | CG_CAPABILITY_RESIZE;
    return CG_SUCCESS;
}

cg_result CreateDevice(const cg_device_desc* desc, cg_device* output) noexcept
{
    if (output == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    constexpr usize legacyDescSize = offsetof(cg_device_desc, native_window);
    if (!StructVersion(desc, legacyDescSize) || desc->width == 0 || desc->height == 0)
    {
        return desc != nullptr
                   ? NotifyCallback(desc->error_callback, desc->error_user_data,
                                    CG_ERROR_INVALID_ARGUMENT, "invalid D3D12 device descriptor")
                   : CG_ERROR_INVALID_ARGUMENT;
    }
    if (desc->size < sizeof(*desc) || desc->native_window == nullptr)
    {
        return NotifyCallback(desc->error_callback, desc->error_user_data, CG_ERROR_UNSUPPORTED,
                              "D3D12 renderer requires the native-window descriptor extension");
    }
    const cyber::rhi::Backend* backend = cyber::rhi::D3D12Backend();
    if (backend == nullptr)
    {
        return NotifyCallback(desc->error_callback, desc->error_user_data, CG_ERROR_UNSUPPORTED,
                              "Direct3D 12 is not available in this build");
    }
    auto* device = AllocateObject<cg_device_t>();
    if (device == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    device->errorCallback = desc->error_callback;
    device->errorUserData = desc->error_user_data;
    {
        std::lock_guard lock(g_mutex);
        device->nextDevice = g_devices;
        g_devices = device;
    }
    DeviceDesc deviceDesc{};
    deviceDesc.backend = backend;
    deviceDesc.adapterIndex = 0;
    auto created = Device::Create(deviceDesc, device->allocator);
    if (!created)
    {
        Notify(device, created.Error(), "Device::Create");
        std::lock_guard lock(g_mutex);
        cg_device* link = &g_devices;
        while (*link != device)
        {
            link = &(*link)->nextDevice;
        }
        *link = device->nextDevice;
        ReleaseObject(device);
        return MapError(created.Error().code);
    }
    device->device = std::move(created.Value());
    cyber::rhi::SwapChainDesc swapchainDesc{};
    swapchainDesc.surface.nativeWindow = desc->native_window;
    swapchainDesc.surface.width = desc->width;
    swapchainDesc.surface.height = desc->height;
    swapchainDesc.format = Format::BGRA8Unorm;
    swapchainDesc.imageCount = 2;
    swapchainDesc.label = "CyberGraphics renderer swapchain";
    auto swapchain = device->device.CreateSwapChain(swapchainDesc);
    if (!swapchain)
    {
        Notify(device, swapchain.Error(), "CreateSwapChain");
        std::lock_guard lock(g_mutex);
        cg_device* link = &g_devices;
        while (*link != device)
        {
            link = &(*link)->nextDevice;
        }
        *link = device->nextDevice;
        ReleaseObject(device);
        return MapError(swapchain.Error().code);
    }
    device->swapchain = std::move(swapchain.Value());
    device->width = desc->width;
    device->height = desc->height;
    if (!CreatePipeline(device))
    {
        std::lock_guard lock(g_mutex);
        cg_device* link = &g_devices;
        while (*link != device)
        {
            link = &(*link)->nextDevice;
        }
        *link = device->nextDevice;
        ReleaseObject(device);
        return CG_ERROR_UNSUPPORTED;
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
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (device->scenes != nullptr || device->meshes != nullptr || device->materials != nullptr)
    {
        return Notify(device, CG_ERROR_INVALID_STATE, "device still owns live resources");
    }
    cg_device* link = &g_devices;
    while (*link != device)
    {
        link = &(*link)->nextDevice;
    }
    *link = device->nextDevice;
    ReleaseObject(device);
    return CG_SUCCESS;
}

cg_result ResizeDevice(cg_device device, cg_uint32 width, cg_uint32 height) noexcept
{
    if (device == nullptr || width == 0 || height == 0)
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid resize extent");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    auto resized = device->swapchain.Resize(width, height);
    if (!resized)
    {
        return Notify(device, resized.Error(), "SwapChain::Resize");
    }
    device->width = width;
    device->height = height;
    return CG_SUCCESS;
}

cg_result CreateScene(cg_device device, const cg_scene_desc* desc, cg_scene* output) noexcept
{
    if (output == nullptr || !StructVersion(desc, sizeof(*desc)))
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid scene descriptor");
    }
    *output = nullptr;
    auto* scene = AllocateObject<cg_scene_t>();
    if (scene == nullptr)
    {
        return Notify(device, CG_ERROR_OUT_OF_MEMORY, "scene allocation failed");
    }
    std::lock_guard lock(g_mutex);
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
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid scene handle");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device) || !ContainsScene(device, scene) || scene->views != nullptr ||
        scene->instances != nullptr)
    {
        return Notify(device, CG_ERROR_INVALID_STATE, "scene has live children or is stale");
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
    if (scene == nullptr || output == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        !FiniteValues(desc->view, 16) || !FiniteValues(desc->projection, 16) ||
        !FiniteValues(desc->viewport, 4) || !FiniteValues(desc->clear_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    auto* view = AllocateObject<cg_view_t>();
    if (view == nullptr)
    {
        return CG_ERROR_OUT_OF_MEMORY;
    }
    std::lock_guard lock(g_mutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr)
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
    if (view == nullptr || !StructVersion(desc, sizeof(*desc)) || !FiniteValues(desc->view, 16) ||
        !FiniteValues(desc->projection, 16) || !FiniteValues(desc->viewport, 4) ||
        !FiniteValues(desc->clear_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_mutex);
    for (cg_device device = g_devices; device != nullptr; device = device->nextDevice)
    {
        for (cg_scene scene = device->scenes; scene != nullptr; scene = scene->next)
        {
            if (ContainsView(scene, view))
            {
                view->desc = *desc;
                return CG_SUCCESS;
            }
        }
    }
    return CG_ERROR_INVALID_ARGUMENT;
}

cg_result DestroyView(cg_scene scene, cg_view view) noexcept
{
    if (scene == nullptr || view == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_mutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr)
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
    if (device == nullptr || output == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        desc->vertices == nullptr || desc->indices == nullptr || desc->vertex_count < 3 ||
        desc->index_count < 3 || desc->index_count % 3 != 0 ||
        desc->vertex_count > std::numeric_limits<usize>::max() / sizeof(Vertex) ||
        desc->index_count > std::numeric_limits<usize>::max() / sizeof(cg_uint32))
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid mesh descriptor");
    }
    *output = nullptr;
    for (cg_uint32 index = 0; index < desc->vertex_count; ++index)
    {
        if (!StructVersion(desc->vertices + index, sizeof(cg_vertex)) ||
            !FiniteValues(desc->vertices[index].position, 3) ||
            !FiniteValues(desc->vertices[index].color, 4))
        {
            return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid mesh vertex");
        }
    }
    for (cg_uint32 index = 0; index < desc->index_count; ++index)
    {
        if (desc->indices[index] >= desc->vertex_count)
        {
            return Notify(device, CG_ERROR_OUT_OF_RANGE, "mesh index exceeds vertex count");
        }
    }
    auto* mesh = AllocateObject<cg_mesh_t>();
    if (mesh == nullptr)
    {
        return Notify(device, CG_ERROR_OUT_OF_MEMORY, "mesh allocation failed");
    }
    auto* vertices = static_cast<Vertex*>(std::malloc(sizeof(Vertex) * desc->vertex_count));
    if (vertices == nullptr)
    {
        ReleaseObject(mesh);
        return Notify(device, CG_ERROR_OUT_OF_MEMORY, "mesh vertex staging allocation failed");
    }
    for (cg_uint32 index = 0; index < desc->vertex_count; ++index)
    {
        std::memcpy(vertices[index].position, desc->vertices[index].position,
                    sizeof(vertices[index].position));
        std::memcpy(vertices[index].color, desc->vertices[index].color,
                    sizeof(vertices[index].color));
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device) ||
        !UploadBuffer(device, BufferUsage::Vertex, vertices, sizeof(Vertex) * desc->vertex_count,
                      mesh->vertices) ||
        !UploadBuffer(device, BufferUsage::Index, desc->indices,
                      sizeof(cg_uint32) * desc->index_count, mesh->indices))
    {
        std::free(vertices);
        if (mesh->vertices.IsValid())
        {
            (void)device->device.DestroyBuffer(mesh->vertices);
        }
        ReleaseObject(mesh);
        return CG_ERROR_INTERNAL;
    }
    std::free(vertices);
    mesh->owner = device;
    mesh->indexCount = desc->index_count;
    mesh->next = device->meshes;
    device->meshes = mesh;
    *output = mesh;
    return CG_SUCCESS;
}

cg_result DestroyMesh(cg_device device, cg_mesh mesh) noexcept
{
    if (device == nullptr || mesh == nullptr)
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid mesh handle");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device) || FindMesh(device, mesh) == nullptr)
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
                return Notify(device, CG_ERROR_INVALID_STATE, "mesh is referenced by an instance");
            }
        }
    }
    cg_mesh* link = &device->meshes;
    while (*link != mesh)
    {
        link = &(*link)->next;
    }
    *link = mesh->next;
    (void)device->device.DestroyBuffer(mesh->vertices);
    (void)device->device.DestroyBuffer(mesh->indices);
    ReleaseObject(mesh);
    return CG_SUCCESS;
}

cg_result CreateMaterial(cg_device device, const cg_material_desc* desc,
                         cg_material* output) noexcept
{
    if (device == nullptr || output == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        !FiniteValues(desc->base_color, 4))
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid material descriptor");
    }
    *output = nullptr;
    auto* material = AllocateObject<cg_material_t>();
    if (material == nullptr)
    {
        return Notify(device, CG_ERROR_OUT_OF_MEMORY, "material allocation failed");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device))
    {
        ReleaseObject(material);
        return CG_ERROR_INVALID_ARGUMENT;
    }
    material->owner = device;
    material->desc = *desc;
    material->next = device->materials;
    device->materials = material;
    *output = material;
    return CG_SUCCESS;
}

cg_result UpdateMaterial(cg_material material, const cg_material_desc* desc) noexcept
{
    if (material == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        !FiniteValues(desc->base_color, 4))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_mutex);
    for (cg_device device = g_devices; device != nullptr; device = device->nextDevice)
    {
        if (FindMaterial(device, material) != nullptr)
        {
            material->desc = *desc;
            return CG_SUCCESS;
        }
    }
    return CG_ERROR_INVALID_ARGUMENT;
}

cg_result DestroyMaterial(cg_device device, cg_material material) noexcept
{
    if (device == nullptr || material == nullptr)
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid material handle");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device) || FindMaterial(device, material) == nullptr)
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
                return Notify(device, CG_ERROR_INVALID_STATE,
                              "material is referenced by an instance");
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
    if (scene == nullptr || output == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        !FiniteValues(desc->transform, 16))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *output = nullptr;
    std::lock_guard lock(g_mutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr || FindMesh(owner->owner, desc->mesh) == nullptr ||
        FindMaterial(owner->owner, desc->material) == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    auto* instance = AllocateObject<cg_instance_t>();
    if (instance == nullptr)
    {
        return Notify(owner->owner, CG_ERROR_OUT_OF_MEMORY, "instance allocation failed");
    }
    instance->desc = *desc;
    constexpr usize constantBufferBytes = 256;
    std::array<Byte, constantBufferBytes> initialConstants{};
    DrawConstants initial{};
    const Mat4 model = ReadMatrix(desc->transform);
    for (usize column = 0; column < 4; ++column)
    {
        for (usize row = 0; row < 4; ++row)
        {
            initial.objectToClip[column * 4 + row] = model[column][row];
        }
    }
    const cg_material material = FindMaterial(owner->owner, desc->material);
    std::memcpy(initial.baseColor, material->desc.base_color, sizeof(initial.baseColor));
    std::memcpy(initialConstants.data(), &initial, sizeof(initial));
    if (!UploadBuffer(owner->owner, BufferUsage::Uniform, initialConstants.data(),
                      initialConstants.size(), instance->constants))
    {
        ReleaseObject(instance);
        return CG_ERROR_INTERNAL;
    }
    cyber::rhi::BufferDesc stagingDesc{};
    stagingDesc.usage = BufferUsage::HostVisible | BufferUsage::CopySource;
    stagingDesc.sizeBytes = constantBufferBytes;
    auto staging = owner->owner->device.CreateBuffer(stagingDesc);
    if (!staging)
    {
        Notify(owner->owner, staging.Error(), "CreateBuffer instance staging");
        (void)owner->owner->device.DestroyBuffer(instance->constants);
        ReleaseObject(instance);
        return MapError(staging.Error().code);
    }
    instance->staging = staging.Value();
    cyber::rhi::BindingResource resource{};
    resource.type = cyber::rhi::BindingType::UniformBuffer;
    resource.buffer = instance->constants;
    resource.size = constantBufferBytes;
    cyber::rhi::BindGroupDesc groupDesc{};
    groupDesc.layout = owner->owner->drawLayout;
    groupDesc.resources = Span<const cyber::rhi::BindingResource>(&resource, 1);
    auto group = owner->owner->device.CreateBindGroup(groupDesc);
    if (!group)
    {
        Notify(owner->owner, group.Error(), "CreateBindGroup instance");
        (void)owner->owner->device.DestroyBuffer(instance->staging);
        (void)owner->owner->device.DestroyBuffer(instance->constants);
        ReleaseObject(instance);
        return MapError(group.Error().code);
    }
    instance->bindGroup = group.Value();
    instance->owner = scene;
    instance->next = scene->instances;
    scene->instances = instance;
    *output = instance;
    return CG_SUCCESS;
}

cg_result UpdateInstance(cg_instance instance, const cg_instance_desc* desc) noexcept
{
    if (instance == nullptr || !StructVersion(desc, sizeof(*desc)) ||
        !FiniteValues(desc->transform, 16))
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_mutex);
    for (cg_device device = g_devices; device != nullptr; device = device->nextDevice)
    {
        for (cg_scene scene = device->scenes; scene != nullptr; scene = scene->next)
        {
            if (ContainsInstance(scene, instance) && FindMesh(device, desc->mesh) != nullptr &&
                FindMaterial(device, desc->material) != nullptr)
            {
                instance->desc = *desc;
                return CG_SUCCESS;
            }
        }
    }
    return CG_ERROR_INVALID_ARGUMENT;
}

cg_result DestroyInstance(cg_scene scene, cg_instance instance) noexcept
{
    if (scene == nullptr || instance == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    std::lock_guard lock(g_mutex);
    cg_scene owner = FindScene(scene);
    if (owner == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    cg_instance* link = &owner->instances;
    while (*link != nullptr && *link != instance)
    {
        link = &(*link)->next;
    }
    if (*link == nullptr)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    *link = instance->next;
    (void)owner->owner->device.DestroyBindGroup(instance->bindGroup);
    (void)owner->owner->device.DestroyBuffer(instance->staging);
    (void)owner->owner->device.DestroyBuffer(instance->constants);
    ReleaseObject(instance);
    return CG_SUCCESS;
}

cg_result SubmitFrame(cg_device device, cg_scene scene, cg_view view) noexcept
{
    if (device == nullptr || scene == nullptr || view == nullptr)
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "invalid frame handles");
    }
    std::lock_guard lock(g_mutex);
    if (!ContainsDevice(device) || !ContainsScene(device, scene) || !ContainsView(scene, view) ||
        view->owner != scene)
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "scene or view does not belong to device");
    }
    const cg_view_desc& viewDesc = view->desc;
    const float* viewport = viewDesc.viewport;
    if (!(viewport[2] > 0.0f) || !(viewport[3] > 0.0f) || viewport[0] < 0.0f ||
        viewport[1] < 0.0f || viewport[0] + viewport[2] > static_cast<float>(device->width) ||
        viewport[1] + viewport[3] > static_cast<float>(device->height))
    {
        return Notify(device, CG_ERROR_INVALID_ARGUMENT, "view viewport exceeds render surface");
    }
    auto acquired = device->swapchain.AcquireNextImage();
    if (!acquired)
    {
        return Notify(device, acquired.Error(), "SwapChain::AcquireNextImage");
    }
    auto createdList = device->device.CreateCommandList(cyber::rhi::CommandListDesc{});
    if (!createdList)
    {
        return Notify(device, createdList.Error(), "CreateCommandList frame");
    }
    CommandList list = std::move(createdList.Value());
    if (!list.Begin())
    {
        return Notify(device, CG_ERROR_INTERNAL, "CommandList::Begin");
    }
    constexpr usize constantBufferBytes = 256;
    for (cg_instance instance = scene->instances; instance != nullptr; instance = instance->next)
    {
        cg_mesh mesh = FindMesh(device, instance->desc.mesh);
        cg_material material = FindMaterial(device, instance->desc.material);
        if (mesh == nullptr || material == nullptr)
        {
            return Notify(device, CG_ERROR_INVALID_STATE, "instance references a stale resource");
        }
        const Mat4 transform = ReadMatrix(viewDesc.projection) * ReadMatrix(viewDesc.view) *
                               ReadMatrix(instance->desc.transform);
        DrawConstants constants{};
        for (usize column = 0; column < 4; ++column)
        {
            for (usize row = 0; row < 4; ++row)
            {
                constants.objectToClip[column * 4 + row] = transform[column][row];
            }
        }
        std::memcpy(constants.baseColor, material->desc.base_color, sizeof(constants.baseColor));
        std::array<Byte, constantBufferBytes> bytes{};
        std::memcpy(bytes.data(), &constants, sizeof(constants));
        auto written = device->device.WriteBuffer(instance->staging, 0,
                                                  Span<const Byte>(bytes.data(), bytes.size()));
        if (!written)
        {
            return Notify(device, written.Error(), "WriteBuffer instance constants");
        }
        if (!instance->stagingReady)
        {
            const BufferBarrier stagingReady{instance->staging, ResourceState::Undefined,
                                             ResourceState::CopySource};
            if (!list.Barrier(Span<const BufferBarrier>(&stagingReady, 1)))
            {
                return Notify(device, CG_ERROR_INTERNAL, "instance staging barrier failed");
            }
        }
        const BufferBarrier toCopy{instance->constants, ResourceState::ConstantBuffer,
                                   ResourceState::CopyDestination};
        const BufferBarrier toRead{instance->constants, ResourceState::CopyDestination,
                                   ResourceState::ConstantBuffer};
        if (!list.Barrier(Span<const BufferBarrier>(&toCopy, 1)) ||
            !list.CopyBuffer(
                BufferCopy{instance->staging, 0, instance->constants, 0, constantBufferBytes}) ||
            !list.Barrier(Span<const BufferBarrier>(&toRead, 1)))
        {
            return Notify(device, CG_ERROR_INTERNAL, "instance constants upload failed");
        }
    }
    const TextureBarrier toTarget{acquired.Value().texture, cyber::rhi::SubresourceRange{},
                                  ResourceState::Present, ResourceState::RenderTarget};
    if (!list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(&toTarget, 1)))
    {
        return Notify(device, CG_ERROR_INTERNAL, "backbuffer render-target transition failed");
    }
    cyber::rhi::ColorAttachment color{};
    color.view = acquired.Value().view;
    color.loadOp = cyber::rhi::LoadOp::Clear;
    color.storeOp = cyber::rhi::StoreOp::Store;
    color.clearValue =
        cyber::rhi::ClearColorValue::Of(viewDesc.clear_color[0], viewDesc.clear_color[1],
                                        viewDesc.clear_color[2], viewDesc.clear_color[3]);
    cyber::rhi::RenderPassDesc pass{};
    pass.colorAttachments = Span<const cyber::rhi::ColorAttachment>(&color, 1);
    if (!list.BeginRenderPass(pass) ||
        !list.SetViewport(
            cyber::rhi::Viewport{viewport[0], viewport[1], viewport[2], viewport[3], 0.0f, 1.0f}) ||
        !list.SetScissor(cyber::rhi::ScissorRect{
            static_cast<u32>(viewport[0]), static_cast<u32>(viewport[1]),
            static_cast<u32>(viewport[2]), static_cast<u32>(viewport[3])}) ||
        !list.SetPipeline(device->pipeline))
    {
        return Notify(device, CG_ERROR_INTERNAL, "render pass setup failed");
    }
    for (cg_instance instance = scene->instances; instance != nullptr; instance = instance->next)
    {
        cg_mesh mesh = FindMesh(device, instance->desc.mesh);
        if (mesh == nullptr)
        {
            return Notify(device, CG_ERROR_INVALID_STATE, "instance references a stale resource");
        }
        if (!list.SetVertexBuffer(0, mesh->vertices) ||
            !list.SetIndexBuffer(mesh->indices, 0, cyber::rhi::IndexFormat::Uint32) ||
            !list.SetBindGroup(cyber::rhi::BindGroupSlot::PerDraw, instance->bindGroup) ||
            !list.DrawIndexed(cyber::rhi::DrawIndexedArguments{mesh->indexCount, 1, 0, 0, 0}))
        {
            return Notify(device, CG_ERROR_INTERNAL, "scene draw recording failed");
        }
    }
    if (!list.EndRenderPass())
    {
        return Notify(device, CG_ERROR_INTERNAL, "CommandList::EndRenderPass");
    }
    const TextureBarrier toPresent{acquired.Value().texture, cyber::rhi::SubresourceRange{},
                                   ResourceState::RenderTarget, ResourceState::Present};
    if (!list.Barrier(Span<const BufferBarrier>{}, Span<const TextureBarrier>(&toPresent, 1)) ||
        !list.End())
    {
        return Notify(device, CG_ERROR_INTERNAL, "backbuffer presentation transition failed");
    }
    const CommandListHandle handle = list.Handle();
    cyber::rhi::SubmitDesc submit{};
    submit.queue = cyber::rhi::QueueKind::Graphics;
    submit.commandLists = Span<const CommandListHandle>(&handle, 1);
    auto submitted = device->device.Submit(submit);
    if (!submitted)
    {
        return Notify(device, submitted.Error(), "Device::Submit");
    }
    for (cg_instance instance = scene->instances; instance != nullptr; instance = instance->next)
    {
        instance->stagingReady = true;
    }
    auto presented = device->swapchain.Present();
    if (!presented)
    {
        return Notify(device, presented.Error(), "SwapChain::Present");
    }
    auto idle = device->device.WaitIdle();
    return idle ? CG_SUCCESS : Notify(device, idle.Error(), "Device::WaitIdle");
}

cg_result GetPluginApi(cg_uint32 requestedVersion, cg_plugin_api* api) noexcept
{
    constexpr usize prefix = offsetof(cg_plugin_api, get_info) + sizeof(cg_plugin_api::get_info);
    if (api == nullptr || api->size < prefix)
    {
        return CG_ERROR_INVALID_ARGUMENT;
    }
    if (requestedVersion != CG_ABI_VERSION)
    {
        return CG_ERROR_INCOMPATIBLE_VERSION;
    }
    cg_plugin_api table{};
    table.size = sizeof(table);
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

    const cg_uint32 capacity = api->size;
    const cg_uint32 tableSize = static_cast<cg_uint32>(sizeof(table));
    const cg_uint32 written = capacity < tableSize ? capacity : tableSize;
    table.size = written;
    std::memcpy(api, &table, written);
    return CG_SUCCESS;
}

}  // namespace

CG_PLUGIN_EXPORT cg_result CG_CALL cgGetPluginApi(cg_uint32 requested_abi_version,
                                                  cg_plugin_api* in_out_api)
{
    return GetPluginApi(requested_abi_version, in_out_api);
}
