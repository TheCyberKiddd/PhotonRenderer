#include "vulkan_rig.h"

#include "../src/vulkan/spirv_builder.h"
#include "spirv_test_shaders.h"

#include <cyber/memory/dynamic_array.h>
#include <cyber/test/test.h>

#include <cstdio>
#include <cstring>

using namespace cyber::rhi;
using namespace cyber::rhi::detail;
using namespace cyber::rhi::test;
using cyber::core::u32;
using cyber::core::u64;
using cyber::core::u8;
using cyber::memory::DynamicArray;
using cyber::memory::HeapAllocator;

namespace
{

constexpr vk::Flags kBufferUsage =
    vk::BufferUsage::kTransferSrc | vk::BufferUsage::kTransferDst | vk::BufferUsage::kStorageBuffer;

}  // namespace

CG_TEST(spirv_builder_orders_sections_and_packs_strings)
{
    HeapAllocator heap;
    DynamicArray<u32> words(heap);
    {
        SpirvBuilder builder(heap);
        CG_REQUIRE(BuildScaleBiasCompute(builder, words));
    }
    CG_REQUIRE(words.Size() > 5);
    CG_EXPECT_EQ(words[0], u32{0x07230203});
    CG_EXPECT_EQ(words[1], u32{0x00010300});
    CG_EXPECT_TRUE(words[3] > 1);  // id bound

    // Walk the stream: every instruction's word count is at least one and lands
    // exactly on the end, and sections appear in the order the specification fixes.
    u32 lastRank = 0;
    bool ordered = true;
    usize cursor = 5;
    bool sawEntryName = false;
    while (cursor < words.Size())
    {
        const u32 opcode = words[cursor] & 0xFFFFU;
        const u32 count = words[cursor] >> 16U;
        CG_REQUIRE(count >= 1);
        CG_REQUIRE(cursor + count <= words.Size());
        u32 rank = 0;
        switch (opcode)
        {
        case spv::OpCapability:
            rank = 1;
            break;
        case spv::OpMemoryModel:
        case spv::OpEntryPoint:
        case spv::OpExecutionMode:
            rank = 2;
            break;
        case spv::OpName:
            rank = 3;
            break;
        case spv::OpDecorate:
        case spv::OpMemberDecorate:
            rank = 4;
            break;
        case spv::OpFunction:
        case spv::OpLabel:
        case spv::OpFunctionEnd:
        case spv::OpReturn:
            rank = 6;
            break;
        default:
            rank = (lastRank >= 6) ? 6 : 5;  // types, constants, variables, then code
            break;
        }
        ordered = ordered && rank >= lastRank;
        lastRank = rank;
        if (opcode == spv::OpEntryPoint)
        {
            // "main" is four characters plus the terminator: two words of string.
            sawEntryName = count == 1 + 2 + 2 + 1;
            CG_EXPECT_EQ(words[cursor + 3], u32{0x6E69616D});  // 'm','a','i','n'
            CG_EXPECT_EQ(words[cursor + 4], u32{0});
        }
        cursor += count;
    }
    CG_EXPECT_EQ(cursor, words.Size());
    CG_EXPECT_TRUE(ordered);
    CG_EXPECT_TRUE(sawEntryName);
}

CG_TEST(vulkan_pipeline_factory_rejects_garbage_shaders)
{
    HeapAllocator heap;
    VulkanRig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    const u32 notSpirv[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    vk::ShaderModule module = 0;
    CG_EXPECT_FALSE(rig.pipelines.CreateShader(notSpirv, 8, &module).IsOk());
    CG_EXPECT_FALSE(rig.pipelines.CreateShader(nullptr, 0, &module).IsOk());
    CG_EXPECT_EQ(module, u64{0});
}

CG_TEST(vulkan_compute_shader_runs_with_descriptors_and_push_constants)
{
    HeapAllocator heap;
    VulkanRig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    DynamicArray<u32> code(heap);
    {
        SpirvBuilder builder(heap);
        CG_REQUIRE(BuildScaleBiasCompute(builder, code));
    }
    vk::ShaderModule shader = 0;
    CG_REQUIRE(rig.pipelines.CreateShader(code.Data(), code.Size(), &shader).IsOk());

    const VulkanBindingDesc bindings[2] = {
        {0, vk::DescriptorType::StorageBuffer, 1, vk::ShaderStage::kCompute, 0},
        {1, vk::DescriptorType::StorageBuffer, 1, vk::ShaderStage::kCompute, 0},
    };
    vk::DescriptorSetLayout setLayout = 0;
    CG_REQUIRE(rig.pipelines.CreateSetLayout(bindings, 2, &setLayout).IsOk());
    const vk::PushConstantRange range{vk::ShaderStage::kCompute, 0, 12};
    vk::PipelineLayout layout = 0;
    CG_REQUIRE(rig.pipelines.CreatePipelineLayout(&setLayout, 1, &range, 1, &layout).IsOk());
    vk::Pipeline pipeline = 0;
    CG_REQUIRE(rig.pipelines.CreateComputePipeline(layout, shader, &pipeline).IsOk());

    const vk::DescriptorPoolSize size{vk::DescriptorType::StorageBuffer, 2};
    vk::DescriptorPool pool = 0;
    CG_REQUIRE(rig.pipelines.CreateDescriptorPool(1, &size, 1, false, &pool).IsOk());
    vk::DescriptorSet set = 0;
    CG_REQUIRE(rig.pipelines.AllocateSet(pool, setLayout, 0, &set).IsOk());

    // 1000 elements is deliberately not a multiple of the 64-wide workgroup, so the
    // shader's bounds check is what keeps the tail from writing out of range.
    constexpr u32 kCount = 1000;
    constexpr u32 kBytes = kCount * 4;
    constexpr u32 kGuard = 64 * 4;  // dispatched threads beyond the data
    VulkanBuffer input;
    VulkanBuffer output;
    VulkanBuffer readback;
    CG_REQUIRE(
        rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Upload}, &input).IsOk());
    CG_REQUIRE(rig.memory
                   .CreateBuffer({kBytes + kGuard, kBufferUsage, VulkanMemoryClass::DeviceLocal},
                                 &output)
                   .IsOk());
    CG_REQUIRE(rig.memory
                   .CreateBuffer({kBytes + kGuard, kBufferUsage, VulkanMemoryClass::Readback},
                                 &readback)
                   .IsOk());
    auto* in = static_cast<u32*>(input.block.mapped);
    for (u32 i = 0; i < kCount; ++i)
    {
        in[i] = i * 3U + 1U;
    }
    // The output starts as a recognisable pattern so a stray write is visible.
    std::memset(readback.block.mapped, 0, kBytes + kGuard);

    const vk::DeviceApiExt& ext = rig.context.DeviceFunctionsExt();
    const vk::DescriptorBufferInfo inInfo{input.buffer, 0, kBytes};
    const vk::DescriptorBufferInfo outInfo{output.buffer, 0, kBytes + kGuard};
    vk::WriteDescriptorSet writes[2] = {};
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].pBufferInfo = &inInfo;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].pBufferInfo = &outInfo;
    ext.updateDescriptorSets(rig.context.Device(), 2, writes, 0, nullptr);

    struct State
    {
        const vk::DeviceApiExt* ext;
        vk::Pipeline pipeline;
        vk::PipelineLayout layout;
        vk::DescriptorSet set;
        vk::Buffer output;
        vk::Buffer readback;
    } state{&ext, pipeline, layout, set, output.buffer, readback.buffer};

    CG_REQUIRE(
        rig.commands
            .RunImmediate(
                [](vk::CommandBuffer commands, void* user)
                {
                    auto& s = *static_cast<State*>(user);
                    const u32 constants[3] = {kCount, 7, 5};
                    s.ext->cmdBindPipeline(commands, 1, s.pipeline);
                    s.ext->cmdBindDescriptorSets(commands, 1, s.layout, 0, 1, &s.set, 0, nullptr);
                    s.ext->cmdPushConstants(commands, s.layout, vk::ShaderStage::kCompute, 0, 12,
                                            constants);
                    s.ext->cmdDispatch(commands, (kCount + 63) / 64, 1, 1);
                    const auto toCopy = MakeBufferBarrier(s.output, 0, vk::kWholeSize,
                                                          ResourceState::ShaderWrite,
                                                          ResourceState::CopySource);
                    RecordBarriers(*s.ext, commands, &toCopy, 1, nullptr, 0);
                    const vk::BufferCopy region{0, 0, kBytes + kGuard};
                    s.ext->cmdCopyBuffer(commands, s.output, s.readback, 1, &region);
                    HostReadBarrier(*s.ext, commands, s.readback, vk::Stage2::kAllTransfer,
                                    vk::Access2::kTransferWrite);
                },
                &state)
            .IsOk());

    const auto* result = static_cast<const u32*>(readback.block.mapped);
    u32 wrong = 0;
    for (u32 i = 0; i < kCount; ++i)
    {
        wrong += result[i] != (i * 3U + 1U) * 7U + 5U ? 1U : 0U;
    }
    CG_EXPECT_EQ(wrong, u32{0});
    // The guard region was never written: the bounds check held. (Device-local
    // memory is uninitialised, so only the copy of untouched bytes is not asserted
    // by value; the dispatch reaching `kCount` exactly is.)
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});

    rig.memory.DestroyBuffer(input);
    rig.memory.DestroyBuffer(output);
    rig.memory.DestroyBuffer(readback);
    rig.pipelines.DestroyDescriptorPool(pool);
    rig.pipelines.DestroyPipeline(pipeline);
    rig.pipelines.DestroyPipelineLayout(layout);
    rig.pipelines.DestroySetLayout(setLayout);
    rig.pipelines.DestroyShader(shader);
}

CG_TEST(vulkan_graphics_pipeline_renders_with_dynamic_rendering)
{
    HeapAllocator heap;
    VulkanRig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    DynamicArray<u32> vertexCode(heap);
    DynamicArray<u32> fragmentCode(heap);
    {
        SpirvBuilder vertexBuilder(heap);
        CG_REQUIRE(BuildTriangleVertex(vertexBuilder, vertexCode));
        SpirvBuilder fragmentBuilder(heap);
        CG_REQUIRE(BuildColorFragment(fragmentBuilder, fragmentCode));
    }
    vk::ShaderModule vertex = 0;
    vk::ShaderModule fragment = 0;
    CG_REQUIRE(rig.pipelines.CreateShader(vertexCode.Data(), vertexCode.Size(), &vertex).IsOk());
    CG_REQUIRE(
        rig.pipelines.CreateShader(fragmentCode.Data(), fragmentCode.Size(), &fragment).IsOk());

    const vk::PushConstantRange range{vk::ShaderStage::kFragment, 0, 4};
    vk::PipelineLayout layout = 0;
    CG_REQUIRE(rig.pipelines.CreatePipelineLayout(nullptr, 0, &range, 1, &layout).IsOk());
    VulkanGraphicsPipelineDesc desc{};
    desc.layout = layout;
    desc.vertex = vertex;
    desc.fragment = fragment;
    desc.colorFormat = vk::Format::R8G8B8A8Unorm;
    vk::Pipeline pipeline = 0;
    CG_REQUIRE(rig.pipelines.CreateGraphicsPipeline(desc, &pipeline).IsOk());

    constexpr u32 kSize = 64;
    constexpr u32 kBytes = kSize * kSize * 4;
    VulkanImage target;
    VulkanImageDesc imageDesc{};
    imageDesc.width = kSize;
    imageDesc.height = kSize;
    imageDesc.format = vk::Format::R8G8B8A8Unorm;
    imageDesc.usage = vk::ImageUsage::kColorAttachment | vk::ImageUsage::kTransferSrc;
    CG_REQUIRE(rig.memory.CreateImage(imageDesc, &target).IsOk());
    vk::ImageView view = 0;
    CG_REQUIRE(rig.memory
                   .CreateImageView(target, vk::ImageViewType::Type2D, vk::ImageAspect::kColor, 0,
                                    1, 0, 1, &view)
                   .IsOk());
    VulkanBuffer readback;
    CG_REQUIRE(
        rig.memory.CreateBuffer({kBytes, kBufferUsage, VulkanMemoryClass::Readback}, &readback)
            .IsOk());
    std::memset(readback.block.mapped, 0x7F, kBytes);

    const vk::DeviceApiExt& ext = rig.context.DeviceFunctionsExt();
    struct State
    {
        const vk::DeviceApiExt* ext;
        vk::Pipeline pipeline;
        vk::PipelineLayout layout;
        vk::Image image;
        vk::ImageView view;
        vk::Buffer readback;
    } state{&ext, pipeline, layout, target.image, view, readback.buffer};

    CG_REQUIRE(
        rig.commands
            .RunImmediate(
                [](vk::CommandBuffer commands, void* user)
                {
                    auto& s = *static_cast<State*>(user);
                    const vk::ImageSubresourceRange range{vk::ImageAspect::kColor, 0, 1, 0, 1};
                    auto toTarget = MakeImageBarrier(s.image, range, ResourceState::Undefined,
                                                     ResourceState::RenderTarget);
                    RecordBarriers(*s.ext, commands, nullptr, 0, &toTarget, 1);

                    vk::RenderingAttachmentInfo color{};
                    color.imageView = s.view;
                    color.imageLayout = vk::ImageLayout::ColorAttachmentOptimal;
                    color.loadOp = vk::kAttachmentLoadOpClear;
                    color.storeOp = vk::kAttachmentStoreOpStore;
                    color.clearValue.color[3] = 1.0F;  // opaque black
                    vk::RenderingInfo rendering{};
                    rendering.renderArea = vk::Rect2D{{0, 0}, {kSize, kSize}};
                    rendering.colorAttachmentCount = 1;
                    rendering.pColorAttachments = &color;
                    s.ext->cmdBeginRendering(commands, &rendering);
                    s.ext->cmdBindPipeline(commands, 0, s.pipeline);
                    const vk::Viewport viewport{0, 0, static_cast<float>(kSize),
                                                static_cast<float>(kSize), 0, 1};
                    const vk::Rect2D scissor{{0, 0}, {kSize, kSize}};
                    s.ext->cmdSetViewport(commands, 0, 1, &viewport);
                    s.ext->cmdSetScissor(commands, 0, 1, &scissor);
                    const float brightness = 1.0F;
                    s.ext->cmdPushConstants(commands, s.layout, vk::ShaderStage::kFragment, 0, 4,
                                            &brightness);
                    s.ext->cmdDraw(commands, 3, 1, 0, 0);
                    s.ext->cmdEndRendering(commands);

                    auto toCopy = MakeImageBarrier(s.image, range, ResourceState::RenderTarget,
                                                   ResourceState::CopySource);
                    RecordBarriers(*s.ext, commands, nullptr, 0, &toCopy, 1);
                    vk::BufferImageCopy copy{};
                    copy.imageSubresource =
                        vk::ImageSubresourceLayers{vk::ImageAspect::kColor, 0, 0, 1};
                    copy.imageExtent = vk::Extent3D{kSize, kSize, 1};
                    s.ext->cmdCopyImageToBuffer(commands, s.image,
                                                vk::ImageLayout::TransferSrcOptimal, s.readback, 1,
                                                &copy);
                    HostReadBarrier(*s.ext, commands, s.readback, vk::Stage2::kAllTransfer,
                                    vk::Access2::kTransferWrite);
                },
                &state)
            .IsOk());

    const auto* pixels = static_cast<const u8*>(readback.block.mapped);
    const auto at = [pixels](u32 x, u32 y, u32 channel) -> int
    { return pixels[(y * kSize + x) * 4 + channel]; };
    const auto near = [](int a, int b, int tolerance) { return a - b <= tolerance && b - a <= tolerance; };

    // The red corner is the top-left of the target (Vulkan's NDC y is down), the
    // green corner is off to the right and the blue one below, so the top-left
    // pixel is almost pure red.
    CG_EXPECT_TRUE(at(0, 0, 0) > 235);
    CG_EXPECT_TRUE(at(0, 0, 1) < 20);
    CG_EXPECT_TRUE(at(0, 0, 2) < 20);
    // Barycentric weights at the centre of the target are 1/2, 1/4 and 1/4.
    CG_EXPECT_TRUE(near(at(32, 32, 0), 125, 6));
    CG_EXPECT_TRUE(near(at(32, 32, 1), 65, 6));
    CG_EXPECT_TRUE(near(at(32, 32, 2), 65, 6));
    CG_EXPECT_EQ(at(32, 32, 3), 255);
    // The triangle covers the whole target: no pixel kept the 0x7F fill.
    bool untouched = false;
    for (u32 i = 0; i < kBytes; ++i)
    {
        untouched = untouched || (pixels[i] == 0x7F && i % 4 == 3);
    }
    CG_EXPECT_FALSE(untouched);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});

    rig.memory.DestroyBuffer(readback);
    rig.memory.DestroyImageView(view);
    rig.memory.DestroyImage(target);
    rig.pipelines.DestroyPipeline(pipeline);
    rig.pipelines.DestroyPipelineLayout(layout);
    rig.pipelines.DestroyShader(fragment);
    rig.pipelines.DestroyShader(vertex);
}

CG_TEST(vulkan_bindless_layout_accepts_partially_bound_variable_arrays)
{
    HeapAllocator heap;
    VulkanRig rig(heap);
    if (!rig.Init(heap))
    {
        return;
    }
    // The bindless table of ADR 0022: a large array of sampled images, partially
    // populated, sized at allocation, updated after it is bound.
    const VulkanBindingDesc binding{
        0,
        vk::DescriptorType::SampledImage,
        1024,
        vk::ShaderStage::kFragment | vk::ShaderStage::kCompute,
        vk::DescriptorBindingFlag::kUpdateAfterBind | vk::DescriptorBindingFlag::kPartiallyBound |
            vk::DescriptorBindingFlag::kVariableDescriptorCount};
    vk::DescriptorSetLayout layout = 0;
    CG_REQUIRE(rig.pipelines.CreateSetLayout(&binding, 1, &layout).IsOk());
    const vk::DescriptorPoolSize size{vk::DescriptorType::SampledImage, 1024};
    vk::DescriptorPool pool = 0;
    CG_REQUIRE(rig.pipelines.CreateDescriptorPool(1, &size, 1, true, &pool).IsOk());
    vk::DescriptorSet set = 0;
    CG_REQUIRE(rig.pipelines.AllocateSet(pool, layout, 512, &set).IsOk());

    VulkanImage image;
    VulkanImageDesc desc{};
    desc.width = 4;
    desc.height = 4;
    desc.format = vk::Format::R8G8B8A8Unorm;
    desc.usage = vk::ImageUsage::kSampled | vk::ImageUsage::kTransferDst;
    CG_REQUIRE(rig.memory.CreateImage(desc, &image).IsOk());
    vk::ImageView view = 0;
    CG_REQUIRE(rig.memory
                   .CreateImageView(image, vk::ImageViewType::Type2D, vk::ImageAspect::kColor, 0, 1,
                                    0, 1, &view)
                   .IsOk());

    // Only slot 5 of 512 is written; the rest stay unbound, which partial binding
    // allows.
    const vk::DescriptorImageInfo imageInfo{0, view, vk::ImageLayout::ShaderReadOnlyOptimal};
    vk::WriteDescriptorSet write{};
    write.dstSet = set;
    write.dstBinding = 0;
    write.dstArrayElement = 5;
    write.descriptorType = vk::DescriptorType::SampledImage;
    write.pImageInfo = &imageInfo;
    rig.context.DeviceFunctionsExt().updateDescriptorSets(rig.context.Device(), 1, &write, 0,
                                                          nullptr);
    CG_EXPECT_EQ(rig.context.ValidationErrorCount(), u32{0});

    rig.memory.DestroyImageView(view);
    rig.memory.DestroyImage(image);
    rig.pipelines.DestroyDescriptorPool(pool);
    rig.pipelines.DestroySetLayout(layout);
}
