#include <cyber/rhi/backend_select.h>
#include <cyber/rhi/d3d12.h>
#include <cyber/rhi/vulkan.h>

#include <cstdlib>
#include <cstring>

namespace cyber::rhi
{

namespace
{

bool EqualsIgnoreCase(const char* a, const char* b) noexcept
{
    while (*a != '\0' && *b != '\0')
    {
        const char x = (*a >= 'A' && *a <= 'Z') ? static_cast<char>(*a - 'A' + 'a') : *a;
        const char y = (*b >= 'A' && *b <= 'Z') ? static_cast<char>(*b - 'A' + 'a') : *b;
        if (x != y)
        {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

constexpr const char kArgument[] = "--backend=";

}  // namespace

const Backend* BackendByName(const char* name) noexcept
{
    if (name == nullptr)
    {
        return nullptr;
    }
    if (EqualsIgnoreCase(name, "d3d12"))
    {
        return D3D12Backend();
    }
    if (EqualsIgnoreCase(name, "vulkan"))
    {
        return VulkanBackend();
    }
    if (EqualsIgnoreCase(name, "null"))
    {
        return NullBackend();
    }
    return nullptr;
}

bool IsBackendArgument(const char* argument) noexcept
{
    return argument != nullptr && std::strncmp(argument, kArgument, sizeof(kArgument) - 1) == 0;
}

bool ReadEnvironmentVariable(const char* name, char* out, cyber::core::usize capacity) noexcept
{
    if (name == nullptr || out == nullptr || capacity == 0)
    {
        return false;
    }
    out[0] = '\0';
#if defined(_MSC_VER)
    cyber::core::usize length = 0;
    if (getenv_s(&length, out, capacity, name) != 0 || length == 0)
    {
        out[0] = '\0';
        return false;
    }
    return true;
#else
    const char* value = std::getenv(name);
    if (value == nullptr)
    {
        return false;
    }
    const cyber::core::usize length = std::strlen(value);
    if (length + 1 > capacity)
    {
        return false;
    }
    std::memcpy(out, value, length + 1);
    return true;
#endif
}

const Backend* SelectBackend(int argc, const char* const* argv, const Backend* fallback,
                             const char** error) noexcept
{
    const char* name = nullptr;
    for (int index = 1; argv != nullptr && index < argc; ++index)
    {
        if (IsBackendArgument(argv[index]))
        {
            name = argv[index] + sizeof(kArgument) - 1;
        }
    }
    char environment[32] = {};
    if (name == nullptr &&
        ReadEnvironmentVariable("CYBER_RHI_BACKEND", environment, sizeof(environment)) &&
        environment[0] != '\0')
    {
        name = environment;
    }
    if (name == nullptr)
    {
        return fallback;
    }
    const Backend* backend = BackendByName(name);
    if (backend == nullptr && error != nullptr)
    {
        *error = "unknown or unavailable backend (expected d3d12, vulkan or null)";
    }
    return backend;
}

}  // namespace cyber::rhi
