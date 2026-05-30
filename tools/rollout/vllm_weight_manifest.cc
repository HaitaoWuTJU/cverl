#include "cverl/model/hf_model_loader.h"
#include "cverl/model/qwen3_5_text.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

std::string arg_value(int& i, int argc, char** argv, const char* name) {
  if (i + 1 >= argc) {
    throw std::invalid_argument(std::string(name) + " requires a value");
  }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string model_dir = "../models/Qwen3.5-0.8B";
    std::string send_dtype = "float32";
    bool packed = true;
    int64_t packed_buffer_size_bytes = 1024LL * 1024LL * 1024LL;
    int64_t packed_num_buffers = 2;

    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--model-dir") {
        model_dir = arg_value(i, argc, argv, "--model-dir");
      } else if (a == "--send-dtype") {
        send_dtype = arg_value(i, argc, argv, "--send-dtype");
      } else if (a == "--packed") {
        packed = true;
      } else if (a == "--no-packed") {
        packed = false;
      } else if (a == "--packed-buffer-size-bytes") {
        packed_buffer_size_bytes = std::stoll(arg_value(i, argc, argv, "--packed-buffer-size-bytes"));
      } else if (a == "--packed-num-buffers") {
        packed_num_buffers = std::stoll(arg_value(i, argc, argv, "--packed-num-buffers"));
      } else {
        throw std::invalid_argument("unknown argument: " + a);
      }
    }
    if (send_dtype != "float32" && send_dtype != "bfloat16" && send_dtype != "float16") {
      throw std::invalid_argument("--send-dtype must be float32|bfloat16|float16");
    }

    cverl::HfModelLoader loader(model_dir);
    cverl::Qwen35TextModel model(std::move(loader));
    auto names = model.required_weight_names(/*max_layers=*/-1);

    nlohmann::json update_info;
    update_info["names"] = nlohmann::json::array();
    update_info["dtype_names"] = nlohmann::json::array();
    update_info["shapes"] = nlohmann::json::array();
    update_info["packed"] = packed;
    update_info["packed_buffer_size_bytes"] = packed_buffer_size_bytes;
    update_info["packed_num_buffers"] = packed_num_buffers;
    update_info["is_checkpoint_format"] = false;

    for (const auto& name : names) {
      auto tensor = model.loader().load_tensor(name);
      update_info["names"].push_back(name);
      update_info["dtype_names"].push_back(send_dtype);
      update_info["shapes"].push_back(tensor.sizes().vec());
    }

    nlohmann::json out;
    out["backend"] = "vllm_native_rl_nccl";
    out["model_dir"] = model_dir;
    out["update_info"] = std::move(update_info);
    std::cout << out.dump(2) << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "vllm_weight_manifest failed: " << e.what() << "\n";
    return 1;
  }
}
