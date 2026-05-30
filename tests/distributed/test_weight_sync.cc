#include "cverl/distributed/weight_sync.h"

#include <torch/torch.h>

#include <iostream>
#include <stdexcept>

namespace {

void require(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

struct TinyModule : torch::nn::Module {
  TinyModule() {
    w = register_parameter("w", torch::arange(6, torch::kFloat32).reshape({2, 3}).contiguous());
    b = register_parameter("b", torch::ones({3}, torch::kFloat32).contiguous());
  }
  torch::Tensor w;
  torch::Tensor b;
};

}  // namespace

int main() {
  try {
    TinyModule module;
    cverl::distributed::SingleProcessCollectives comm;
    auto params = cverl::distributed::module_parameter_views(module);
    cverl::distributed::broadcast_parameters_from_root(params, comm, 0, {0});

    require(torch::allclose(module.w, torch::arange(6, torch::kFloat32).reshape({2, 3})), "w unchanged");
    require(torch::allclose(module.b, torch::ones({3}, torch::kFloat32)), "b unchanged");

    bool rejected = false;
    try {
      cverl::distributed::broadcast_parameters_from_root(params, comm, 1, {0});
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "invalid root rejected");

    std::cout << "weight sync tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "test_weight_sync failed: " << e.what() << "\n";
    return 1;
  }
}
