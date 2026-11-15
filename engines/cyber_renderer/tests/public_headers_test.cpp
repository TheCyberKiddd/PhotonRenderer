#include <cyber/renderer/bindless_table.h>
#include <cyber/renderer/clustered_lighting.h>
#include <cyber/renderer/descriptor_allocator.h>
#include <cyber/renderer/frame_stream.h>
#include <cyber/renderer/gpu_culling_pipeline.h>
#include <cyber/renderer/gpu_driven.h>
#include <cyber/renderer/gpu_light_culling.h>
#include <cyber/renderer/gpu_texture_pool.h>
#include <cyber/renderer/gpu_virtual_texture.h>
#include <cyber/renderer/gpu_virtual_texture_feedback.h>
#include <cyber/renderer/hot_pipeline.h>
#include <cyber/renderer/material.h>
#include <cyber/renderer/meshlet.h>
#include <cyber/renderer/raytracing_path.h>
#include <cyber/renderer/reflected_layout.h>
#include <cyber/renderer/render_graph.h>
#include <cyber/renderer/render_thread.h>
#include <cyber/renderer/texture_streaming.h>
#include <cyber/renderer/virtual_texture.h>
#include <cyber/test/test.h>

#include <type_traits>

using namespace cyber::renderer;

CG_TEST(renderer_public_headers)
{
    static_assert(std::is_trivially_copyable_v<ReflectedLayout>);
    static_assert(std::is_trivially_copyable_v<ReflectedVertexLayout>);
    static_assert(std::is_move_constructible_v<MaterialDefinition>);
    static_assert(!std::is_copy_constructible_v<MaterialInstance>);
    static_assert(std::is_move_constructible_v<MaterialInstance>);
    static_assert(!std::is_copy_constructible_v<HotGraphicsPipeline>);
    static_assert(std::is_move_constructible_v<HotGraphicsPipeline>);
    static_assert(!std::is_copy_constructible_v<FrameBufferStream>);
    static_assert(std::is_move_constructible_v<FrameBufferStream>);
    static_assert(!std::is_copy_constructible_v<PersistentDescriptorAllocator>);
    static_assert(std::is_move_constructible_v<PersistentDescriptorAllocator>);
    static_assert(!std::is_copy_constructible_v<TransientDescriptorAllocator>);
    static_assert(std::is_move_constructible_v<TransientDescriptorAllocator>);
    static_assert(!std::is_copy_constructible_v<BindlessResourceTable>);
    static_assert(std::is_move_constructible_v<BindlessResourceTable>);
    static_assert(!std::is_copy_constructible_v<ClusterLightLists>);
    static_assert(std::is_move_constructible_v<ClusterLightLists>);
    static_assert(!std::is_copy_constructible_v<RenderGraph>);
    static_assert(!std::is_move_constructible_v<RenderGraph>);
    static_assert(!std::is_copy_constructible_v<RenderThread>);
    static_assert(std::is_move_constructible_v<RenderThread>);

    CG_EXPECT_EQ(kDefaultRenderFramesInFlight, u32{2});
    CG_EXPECT_EQ(kMaxReflectedPushConstantBytes, u32{128});
    CG_EXPECT_EQ(kMaxMaterialParameters, u32{64});
    CG_EXPECT_EQ(SelectResourceBindingPath(cyber::rhi::Device{}),
                 ResourceBindingPath::Conventional);
    CG_EXPECT_EQ(ToRhiStage(cyber::shaders::Stage::Pixel), cyber::rhi::ShaderStage::Pixel);

    HotGraphicsPipeline empty;
    CG_EXPECT_FALSE(empty.Current().IsValid());
    CG_EXPECT_FALSE(empty.HasBuilt());
    CG_EXPECT_EQ(empty.Layout().bindingCounts[0], u32{0});
    CG_EXPECT_TRUE(empty.LastError() == nullptr);
}

CG_TEST(renderer_public_headers_raytracing_path)
{
    static_assert(std::is_trivially_copyable_v<RayTracingPlan>);
    static_assert(std::is_trivially_copyable_v<RayTracingPolicyDesc>);
    const RayTracingPlan plan{};
    CG_EXPECT_FALSE(plan.UsesHardwareRayTracing());
    const RayTracingSelection shadows = plan.For(RayTracedEffect::Shadows);
    CG_EXPECT_EQ(shadows.technique, RayTracingTechnique::Rasterized);
}

CG_TEST(renderer_public_headers_gpu_driven)
{
    static_assert(std::is_trivially_copyable_v<GpuCullInstance>);
    static_assert(std::is_trivially_copyable_v<GpuDrawGroup>);
    static_assert(std::is_trivially_copyable_v<GpuDrawBatch>);
    static_assert(std::is_trivially_copyable_v<GpuCullCounters>);
    static_assert(std::is_trivially_copyable_v<GpuCullView>);
    static_assert(std::is_trivially_copyable_v<GeometryPathPlan>);
    static_assert(std::is_trivially_copyable_v<AsyncComputePlan>);
    static_assert(std::is_trivially_copyable_v<MeshletBounds>);
    static_assert(!std::is_copy_constructible_v<GpuDrivenScene>);
    static_assert(std::is_move_constructible_v<GpuDrivenScene>);
    static_assert(!std::is_copy_constructible_v<HiZPyramid>);
    static_assert(std::is_move_constructible_v<HiZPyramid>);
    static_assert(!std::is_copy_constructible_v<GpuCullingPipeline>);
    static_assert(std::is_move_constructible_v<GpuCullingPipeline>);
    static_assert(!std::is_copy_constructible_v<GpuLightCullingPass>);
    static_assert(std::is_move_constructible_v<GpuLightCullingPass>);

    CG_EXPECT_EQ(kGpuDrivenInstanceStreamBinding, u32{1});
    CG_EXPECT_EQ(kMeshletMaxVertices, u32{64});
    CG_EXPECT_EQ(kMeshletMaxTriangles, u32{124});
    const GpuCullingPipeline emptyPipeline;
    CG_EXPECT_FALSE(emptyPipeline.IsValid());
    CG_EXPECT_EQ(emptyPipeline.BatchCount(), u32{0});
    const GpuLightCullingPass emptyLights;
    CG_EXPECT_FALSE(emptyLights.IsValid());
    CG_EXPECT_EQ(emptyLights.ClusterCount(), u32{0});
}
