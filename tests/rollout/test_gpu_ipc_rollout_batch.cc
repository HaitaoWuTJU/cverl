#include "cverl/rollout/gpu_ipc_rollout_batch.h"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

int child_main(const std::string& manifest_path, const std::string& output_path) {
  std::ifstream in(manifest_path);
  require(static_cast<bool>(in), "failed to open manifest");
  std::string hex;
  int64_t rows = 0;
  int64_t cols = 0;
  int32_t device = 0;
  in >> hex >> rows >> cols >> device;

  cverl::rollout::GpuIpcTensorHandle handle;
  handle.kind = cverl::rollout::GpuRolloutTensorKind::ResponseIds;
  handle.dtype = cverl::rollout::GpuIpcDType::Int64;
  handle.device_index = device;
  handle.ndim = 2;
  handle.sizes = {rows, cols, 0, 0};
  handle.strides = {cols, 1, 0, 0};
  handle.numel = static_cast<uint64_t>(rows * cols);
  cverl::rollout::cuda_ipc_handle_from_hex(hex, &handle);

  auto imported = cverl::rollout::import_cuda_ipc_tensor(handle, device);
  auto cpu = imported.to(torch::kCPU);
  int64_t sum = cpu.sum().item<int64_t>();
  std::ofstream out(output_path);
  out << sum << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 4 && std::string(argv[1]) == "--child") {
      return child_main(argv[2], argv[3]);
    }

    if (!torch::cuda::is_available()) {
      std::cout << "CUDA unavailable; skipping CUDA IPC rollout batch test\n";
      return 0;
    }
    torch::Device device(torch::kCUDA, 0);
    auto tensor = torch::arange(0, 12, torch::TensorOptions().dtype(torch::kInt64).device(device)).view({3, 4}).contiguous();
    auto handle = cverl::rollout::export_cuda_ipc_tensor(tensor, cverl::rollout::GpuRolloutTensorKind::ResponseIds);

    const auto base = std::filesystem::temp_directory_path() / ("cverl_gpu_ipc_" + std::to_string(::getpid()));
    const auto manifest_path = base.string() + ".manifest";
    const auto output_path = base.string() + ".out";
    {
      std::ofstream manifest(manifest_path);
      manifest << cverl::rollout::cuda_ipc_handle_to_hex(handle) << " "
               << tensor.size(0) << " "
               << tensor.size(1) << " "
               << tensor.get_device() << "\n";
    }

    std::string cmd = std::string("\"") + argv[0] + "\" --child \"" + manifest_path + "\" \"" + output_path + "\"";
    int rc = std::system(cmd.c_str());
    require(rc == 0, "child CUDA IPC import failed");

    std::ifstream out(output_path);
    int64_t got = -1;
    out >> got;
    require(got == 66, "unexpected CUDA IPC imported tensor sum");
    std::filesystem::remove(manifest_path);
    std::filesystem::remove(output_path);
    std::cout << "GPU IPC rollout batch test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_gpu_ipc_rollout_batch failed: " << e.what() << "\n";
    return 1;
  }
}
