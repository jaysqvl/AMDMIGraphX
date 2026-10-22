// Compile an explicit convolution through rocMLIR's public Miir C API.
// No kernel is loaded or executed. Constants for ELF flags come from LLVM ELF.h.
#include <Miir.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc != 4 or std::filesystem::exists("/dev/kfd") or
        std::filesystem::exists("/dev/dri"))
        {
        return 2;
    }
    const std::string arch = argv[1];
    const std::string type = argv[2];
    // LLVM ELF.h, processor-selection values, not ASIC marketing names.
    const std::map<std::string, uint32_t> targets = {
        {"gfx900", 0x2c}, {"gfx906", 0x2f}, {"gfx908", 0x30},
        {"gfx90a", 0x3f}, {"gfx90c", 0x32}, {"gfx942", 0x4c},
        {"gfx950", 0x4f}, {"gfx1030", 0x36}, {"gfx1100", 0x41},
        {"gfx1101", 0x46}, {"gfx1200", 0x48}, {"gfx1201", 0x4e}};
    if (targets.count(arch) == 0 or (type != "fp32" and type != "fp16"))
    {
        return 2;
    }
    const std::string options =
        "--operation conv --arch amdgcn-amd-amdhsa:" + arch +
        " --num_cu 8 --in_type " + type + " --fil_type " + type +
        " --out_type " + type +
        " --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW"
        " --batchsize 1 --groupsize 1 --in_channels 32 --out_channels 32"
        " --in_h 32 --in_w 32 --out_h 32 --out_w 32 --fil_h 3 --fil_w 3"
        " --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1"
        " --padding_h 1 --padding_w 1 --kernel_id 0";
    MiirHandle handle = miirCreateHandle(options.c_str());
    if (!handle)
    {
        std::cerr << "Explicit-target convolution was rejected\n";
        return 3;
    }
    if (miirGetKernelCount(handle) != 1 or miirLowerBin(handle) != MIIR_SUCCESS)
    {
        miirDestroyHandle(handle);
        std::cerr << "Offline convolution lowering failed\n";
        return 4;
    }
    size_t size = 0;
    if (miirBufferGet(handle, nullptr, &size) != MIIR_SUCCESS or size < 64)
    {
        miirDestroyHandle(handle);
        return 5;
    }
    std::vector<char> code(size);
    if (miirBufferGet(handle, code.data(), &size) != MIIR_SUCCESS)
    {
        miirDestroyHandle(handle);
        return 6;
    }
    miirDestroyHandle(handle);
    uint16_t machine = 0;
    uint32_t flags = 0;
    std::memcpy(&machine, code.data() + 18, sizeof(machine));
    std::memcpy(&flags, code.data() + 48, sizeof(flags));
    const uint32_t expected = targets.at(arch);
    if (std::memcmp(code.data(), "\177ELF", 4) != 0 or
        machine != 224 or (flags & 0xff) != expected)
        {
        std::cerr << "Unexpected AMD code object target\n";
        return 7;
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(code.data(), size);
    if (!output)
    {
        return 8;
    }
    std::cout << "{\"offline_mlir_convolution\":\"passed\",\"architecture\":\""
              << arch << "\",\"type\":\"" << type << "\",\"code_bytes\":"
              << size << ",\"elf_machine\":" << machine
              << ",\"elf_architecture_code\":" << (flags & 0xff)
              << ",\"gpu_execution\":false}\n";
    return 0;
}
