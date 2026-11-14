#include <cyber/software/ibl.h>

#include <cmath>

namespace cyber::software
{

core::Result<Cubemap, core::Error> CaptureReflectionProbe(const ReflectionProbeCaptureDesc& desc,
                                                          ReflectionProbeCaptureFn renderFace,
                                                          void* userData,
                                                          memory::Allocator& allocator) noexcept
{
    using ResultType = core::Result<Cubemap, core::Error>;
    if (renderFace == nullptr || desc.faceSize < 1 || desc.faceSize > kMaxCubemapFaceSize ||
        desc.supersampleFactor < kMinSupersampleFactor ||
        desc.supersampleFactor > kMaxSupersampleFactor ||
        desc.faceSize > 16384 / desc.supersampleFactor || !math::IsFinite(desc.position.x) ||
        !math::IsFinite(desc.position.y) || !math::IsFinite(desc.position.z) ||
        !math::IsFinite(desc.nearPlane) || !math::IsFinite(desc.farPlane) ||
        desc.nearPlane <= 0.0f || desc.farPlane <= desc.nearPlane)
    {
        return ResultType::Err(
            core::Error{core::ErrorCode::InvalidArgument, "invalid reflection probe capture"});
    }

    CubemapDesc cubemapDesc{};
    cubemapDesc.faceSize = desc.faceSize;
    auto cubemapResult = Cubemap::Create(cubemapDesc, allocator);
    if (!cubemapResult)
    {
        return ResultType::Err(cubemapResult.Error());
    }
    Cubemap cubemap = std::move(cubemapResult).Value();

    const i32 captureSize = desc.faceSize * desc.supersampleFactor;
    FramebufferDesc colorDesc{};
    colorDesc.width = captureSize;
    colorDesc.height = captureSize;
    colorDesc.format = PixelFormat::RGBA32F;
    auto colorResult = Framebuffer::Create(colorDesc, allocator);
    if (!colorResult)
    {
        return ResultType::Err(colorResult.Error());
    }
    Framebuffer color = std::move(colorResult).Value();

    DepthBufferDesc depthDesc{};
    depthDesc.width = captureSize;
    depthDesc.height = captureSize;
    auto depthResult = DepthBuffer::Create(depthDesc, allocator);
    if (!depthResult)
    {
        return ResultType::Err(depthResult.Error());
    }
    DepthBuffer depth = std::move(depthResult).Value();

    Framebuffer resolved;
    if (desc.supersampleFactor > 1)
    {
        FramebufferDesc resolvedDesc{};
        resolvedDesc.width = desc.faceSize;
        resolvedDesc.height = desc.faceSize;
        resolvedDesc.format = PixelFormat::RGBA32F;
        auto resolvedResult = Framebuffer::Create(resolvedDesc, allocator);
        if (!resolvedResult)
        {
            return ResultType::Err(resolvedResult.Error());
        }
        resolved = std::move(resolvedResult).Value();
    }

    constexpr f32 halfPi = cyber::math::kPi * 0.5f;
    const math::Mat4 projection =
        math::Mat4::PerspectiveReverseZ(halfPi, 1.0f, desc.nearPlane, desc.farPlane);
    for (u32 face = 0; face < kCubeFaceCount; ++face)
    {
        const math::Vec3 forward = CubeDirectionFromFaceUv(face, 0.5f, 0.5f);
        const math::Vec3 top = CubeDirectionFromFaceUv(face, 0.5f, 0.0f);
        const math::Vec3 up =
            cyber::math::Normalize(top - forward * cyber::math::Dot(top, forward));

        ReflectionProbeCaptureView view{};
        view.face = static_cast<CubeFace>(face);
        view.position = desc.position;
        view.view = math::Mat4::LookAt(desc.position, desc.position + forward, up);
        view.projection = projection;

        color.Clear(math::Color{0.0f, 0.0f, 0.0f, 1.0f});
        depth.Clear();
        const auto rendered = renderFace(view, color, depth, userData);
        if (!rendered)
        {
            return ResultType::Err(rendered.Error());
        }
        if (!color.IsValid() || color.Width() != captureSize || color.Height() != captureSize ||
            color.Format() != PixelFormat::RGBA32F || !depth.IsValid() ||
            depth.Width() != captureSize || depth.Height() != captureSize)
        {
            return ResultType::Err(
                core::Error{core::ErrorCode::Internal,
                            "reflection probe callback invalidated a capture target"});
        }

        const Framebuffer* faceColor = &color;
        if (desc.supersampleFactor > 1)
        {
            ResolveDesc resolveDesc{};
            resolveDesc.factor = desc.supersampleFactor;
            resolveDesc.filter = desc.resolveFilter;
            resolveDesc.useJobs = false;
            const auto resolveStatus =
                ResolveSupersampled(color, resolved, resolveDesc, nullptr, allocator);
            if (!resolveStatus)
            {
                return ResultType::Err(resolveStatus.Error());
            }
            faceColor = &resolved;
        }

        for (i32 y = 0; y < desc.faceSize; ++y)
        {
            for (i32 x = 0; x < desc.faceSize; ++x)
            {
                cubemap.SetTexel(face, 0, x, y, faceColor->GetPixel(x, y));
            }
        }
    }

    cubemap.GenerateMips();
    return ResultType::Ok(std::move(cubemap));
}

}  // namespace cyber::software
