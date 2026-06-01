#include <torch/torch.h>
#include <torch/cuda.h>

#include <ATen/Context.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string data_dir = "data/cifar-10-batches-bin";
  std::string device = "cuda:0";
  std::string metrics_csv;
  std::string summary_json;
  int64_t epochs = 10;
  int64_t batch_size = 256;
  int64_t warmup_batches = 10;
  int64_t seed = 1234;
  double lr = 0.1;
  double momentum = 0.9;
  double weight_decay = 5.0e-4;
  bool allow_tf32 = true;
};

std::string arg_str(int argc, char** argv, const std::string& flag, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int64_t arg_i64(int argc, char** argv, const std::string& flag, int64_t fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stoll(argv[i + 1]);
    }
  }
  return fallback;
}

double arg_f64(int argc, char** argv, const std::string& flag, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      return std::stod(argv[i + 1]);
    }
  }
  return fallback;
}

bool arg_bool(int argc, char** argv, const std::string& flag, bool fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == flag) {
      const std::string value = argv[i + 1];
      if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
      }
      if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
      }
      throw std::invalid_argument("invalid boolean for " + flag);
    }
  }
  return fallback;
}

Args parse_args(int argc, char** argv) {
  Args args;
  args.data_dir = arg_str(argc, argv, "--data-dir", args.data_dir);
  args.device = arg_str(argc, argv, "--device", args.device);
  args.metrics_csv = arg_str(argc, argv, "--metrics-csv", args.metrics_csv);
  args.summary_json = arg_str(argc, argv, "--summary-json", args.summary_json);
  args.epochs = arg_i64(argc, argv, "--epochs", args.epochs);
  args.batch_size = arg_i64(argc, argv, "--batch-size", args.batch_size);
  args.warmup_batches = arg_i64(argc, argv, "--warmup-batches", args.warmup_batches);
  args.seed = arg_i64(argc, argv, "--seed", args.seed);
  args.lr = arg_f64(argc, argv, "--lr", args.lr);
  args.momentum = arg_f64(argc, argv, "--momentum", args.momentum);
  args.weight_decay = arg_f64(argc, argv, "--weight-decay", args.weight_decay);
  args.allow_tf32 = arg_bool(argc, argv, "--allow-tf32", args.allow_tf32);
  if (args.epochs <= 0 || args.batch_size <= 0 || args.warmup_batches < 0) {
    throw std::invalid_argument("epochs and batch-size must be positive; warmup-batches must be non-negative");
  }
  return args;
}

struct Cifar10Tensors {
  torch::Tensor images;
  torch::Tensor labels;
};

void read_cifar_file(const std::filesystem::path& path,
                     std::vector<float>* images,
                     std::vector<int64_t>* labels) {
  constexpr int64_t kRecordBytes = 1 + 3 * 32 * 32;
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("missing CIFAR10 file: " + path.string());
  }
  const auto bytes = static_cast<int64_t>(std::filesystem::file_size(path));
  if (bytes <= 0 || bytes % kRecordBytes != 0) {
    throw std::runtime_error("invalid CIFAR10 binary file size: " + path.string());
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open " + path.string());
  }
  std::vector<unsigned char> buffer(static_cast<size_t>(bytes));
  in.read(reinterpret_cast<char*>(buffer.data()), bytes);
  if (in.gcount() != bytes) {
    throw std::runtime_error("short read from " + path.string());
  }
  const int64_t records = bytes / kRecordBytes;
  labels->reserve(labels->size() + static_cast<size_t>(records));
  images->reserve(images->size() + static_cast<size_t>(records * 3 * 32 * 32));
  for (int64_t n = 0; n < records; ++n) {
    const unsigned char* record = buffer.data() + n * kRecordBytes;
    labels->push_back(static_cast<int64_t>(record[0]));
    for (int64_t i = 0; i < 3 * 32 * 32; ++i) {
      images->push_back(static_cast<float>(record[1 + i]) / 255.0f);
    }
  }
}

Cifar10Tensors load_cifar10_split(const std::string& data_dir, const std::string& split) {
  const auto raw_images_path = std::filesystem::path(data_dir) / (split + "_images.f32");
  const auto raw_labels_path = std::filesystem::path(data_dir) / (split + "_labels.i64");
  if (std::filesystem::exists(raw_images_path) && std::filesystem::exists(raw_labels_path)) {
    const int64_t image_bytes = static_cast<int64_t>(std::filesystem::file_size(raw_images_path));
    const int64_t label_bytes = static_cast<int64_t>(std::filesystem::file_size(raw_labels_path));
    constexpr int64_t kImageValues = 3 * 32 * 32;
    if (image_bytes <= 0 || image_bytes % (kImageValues * static_cast<int64_t>(sizeof(float))) != 0 ||
        label_bytes <= 0 || label_bytes % static_cast<int64_t>(sizeof(int64_t)) != 0) {
      throw std::runtime_error("invalid CIFAR10 raw tensor cache sizes in " + data_dir);
    }
    const int64_t n = image_bytes / (kImageValues * static_cast<int64_t>(sizeof(float)));
    if (label_bytes / static_cast<int64_t>(sizeof(int64_t)) != n) {
      throw std::runtime_error("CIFAR10 raw tensor image/label count mismatch in " + data_dir);
    }
    std::vector<float> image_values(static_cast<size_t>(n * kImageValues));
    std::vector<int64_t> label_values(static_cast<size_t>(n));
    std::ifstream image_in(raw_images_path, std::ios::binary);
    std::ifstream label_in(raw_labels_path, std::ios::binary);
    image_in.read(reinterpret_cast<char*>(image_values.data()), image_bytes);
    label_in.read(reinterpret_cast<char*>(label_values.data()), label_bytes);
    if (image_in.gcount() != image_bytes || label_in.gcount() != label_bytes) {
      throw std::runtime_error("short read from CIFAR10 raw tensor cache in " + data_dir);
    }
    auto images = torch::from_blob(image_values.data(), {n, 3, 32, 32}, torch::kFloat32).clone();
    auto labels = torch::from_blob(label_values.data(), {n}, torch::kInt64).clone();
    return {images.contiguous(), labels.contiguous()};
  }

  const auto tensor_images_path = std::filesystem::path(data_dir) / (split + "_images.pt");
  const auto tensor_labels_path = std::filesystem::path(data_dir) / (split + "_labels.pt");
  if (std::filesystem::exists(tensor_images_path) && std::filesystem::exists(tensor_labels_path)) {
    torch::Tensor images;
    torch::Tensor labels;
    torch::load(images, tensor_images_path.string());
    torch::load(labels, tensor_labels_path.string());
    if (!images.defined() || !labels.defined() || images.dim() != 4 || labels.dim() != 1 ||
        images.size(0) != labels.size(0) || images.size(1) != 3 || images.size(2) != 32 || images.size(3) != 32) {
      throw std::runtime_error("invalid CIFAR10 tensor cache in " + data_dir);
    }
    return {images.to(torch::kFloat32).contiguous(), labels.to(torch::kInt64).contiguous()};
  }

  std::vector<float> images;
  std::vector<int64_t> labels;
  if (split == "train") {
    for (int i = 1; i <= 5; ++i) {
      read_cifar_file(std::filesystem::path(data_dir) / ("data_batch_" + std::to_string(i) + ".bin"),
                      &images,
                      &labels);
    }
  } else if (split == "test") {
    read_cifar_file(std::filesystem::path(data_dir) / "test_batch.bin", &images, &labels);
  } else {
    throw std::invalid_argument("unsupported CIFAR10 split: " + split);
  }
  const int64_t n = static_cast<int64_t>(labels.size());
  if (n == 0 || static_cast<int64_t>(images.size()) != n * 3 * 32 * 32) {
    throw std::runtime_error("invalid CIFAR10 " + split + " tensor sizes");
  }
  auto image_tensor = torch::from_blob(images.data(), {n, 3, 32, 32}, torch::kFloat32).clone();
  auto label_tensor = torch::from_blob(labels.data(), {n}, torch::kInt64).clone();
  const auto mean = torch::tensor({0.4914f, 0.4822f, 0.4465f}).view({1, 3, 1, 1});
  const auto std = torch::tensor({0.2470f, 0.2435f, 0.2616f}).view({1, 3, 1, 1});
  image_tensor = (image_tensor - mean) / std;
  return {image_tensor.contiguous(), label_tensor.contiguous()};
}

struct BottleneckImpl : torch::nn::Module {
  static constexpr int64_t kExpansion = 4;

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  torch::nn::BatchNorm2d bn3{nullptr};
  torch::nn::Sequential downsample{nullptr};
  bool has_downsample = false;

  BottleneckImpl(int64_t inplanes, int64_t planes, int64_t stride, bool needs_downsample)
      : conv1(torch::nn::Conv2dOptions(inplanes, planes, 1).bias(false)),
        bn1(planes),
        conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(stride).padding(1).bias(false)),
        bn2(planes),
        conv3(torch::nn::Conv2dOptions(planes, planes * kExpansion, 1).bias(false)),
        bn3(planes * kExpansion),
        has_downsample(needs_downsample) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    if (has_downsample) {
      downsample = torch::nn::Sequential(
          torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes * kExpansion, 1).stride(stride).bias(false)),
          torch::nn::BatchNorm2d(planes * kExpansion));
      register_module("downsample", downsample);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto identity = x;
    auto out = torch::relu(bn1(conv1(x)));
    out = torch::relu(bn2(conv2(out)));
    out = bn3(conv3(out));
    if (has_downsample) {
      identity = downsample->forward(x);
    }
    out = out + identity;
    return torch::relu(out);
  }
};
TORCH_MODULE(Bottleneck);

struct ResNet152CifarImpl : torch::nn::Module {
  int64_t inplanes = 64;
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
  torch::nn::Linear fc{nullptr};

  ResNet152CifarImpl()
      : conv1(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false)),
        bn1(64),
        avgpool(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
        fc(512 * BottleneckImpl::kExpansion, 10) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    layer1 = register_module("layer1", make_layer(64, 3, 1));
    layer2 = register_module("layer2", make_layer(128, 8, 2));
    layer3 = register_module("layer3", make_layer(256, 36, 2));
    layer4 = register_module("layer4", make_layer(512, 3, 2));
    register_module("avgpool", avgpool);
    register_module("fc", fc);
  }

  torch::nn::Sequential make_layer(int64_t planes, int64_t blocks, int64_t stride) {
    torch::nn::Sequential layers;
    const bool needs_downsample = stride != 1 || inplanes != planes * BottleneckImpl::kExpansion;
    layers->push_back(Bottleneck(inplanes, planes, stride, needs_downsample));
    inplanes = planes * BottleneckImpl::kExpansion;
    for (int64_t i = 1; i < blocks; ++i) {
      layers->push_back(Bottleneck(inplanes, planes, 1, false));
    }
    return layers;
  }

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(bn1(conv1(x)));
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);
    x = avgpool(x);
    x = torch::flatten(x, 1);
    return fc(x);
  }
};
TORCH_MODULE(ResNet152Cifar);

void sync_if_cuda(const torch::Device& device) {
  if (device.is_cuda()) {
    torch::cuda::synchronize(device.index());
  }
}

double seconds_since(const std::chrono::steady_clock::time_point& begin) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}

double run_batches(ResNet152Cifar& model,
                   torch::optim::SGD& optimizer,
                   const torch::Tensor& images,
                   const torch::Tensor& labels,
                   int64_t batch_size,
                   int64_t max_batches,
                   bool train) {
  model->train(train);
  const int64_t n = images.size(0);
  const int64_t batches = (n + batch_size - 1) / batch_size;
  const int64_t run_batches = max_batches <= 0 ? batches : std::min<int64_t>(batches, max_batches);
  auto perm = torch::randperm(n, torch::TensorOptions().device(images.device()).dtype(torch::kLong));
  auto loss_sum = torch::zeros({}, torch::TensorOptions().device(images.device()).dtype(torch::kFloat32));
  int64_t seen = 0;
  for (int64_t batch = 0; batch < run_batches; ++batch) {
    const int64_t start = batch * batch_size;
    const int64_t size = std::min<int64_t>(batch_size, n - start);
    auto index = perm.narrow(0, start, size);
    auto x = images.index_select(0, index);
    auto y = labels.index_select(0, index);
    optimizer.zero_grad();
    auto logits = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    loss.backward();
    optimizer.step();
    loss_sum = loss_sum + loss.detach().to(torch::kFloat32) * static_cast<double>(size);
    seen += size;
  }
  return loss_sum.to(torch::kCPU).item<double>() / static_cast<double>(std::max<int64_t>(seen, 1));
}

struct EvalMetrics {
  double loss = 0.0;
  double accuracy = 0.0;
};

EvalMetrics evaluate(ResNet152Cifar& model,
                     const torch::Tensor& images,
                     const torch::Tensor& labels,
                     int64_t batch_size) {
  torch::NoGradGuard no_grad;
  model->eval();
  const int64_t n = images.size(0);
  auto loss_sum = torch::zeros({}, torch::TensorOptions().device(images.device()).dtype(torch::kFloat32));
  auto correct_sum = torch::zeros({}, torch::TensorOptions().device(images.device()).dtype(torch::kInt64));
  for (int64_t start = 0; start < n; start += batch_size) {
    const int64_t size = std::min<int64_t>(batch_size, n - start);
    auto x = images.narrow(0, start, size);
    auto y = labels.narrow(0, start, size);
    auto logits = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    loss_sum = loss_sum + loss.to(torch::kFloat32) * static_cast<double>(size);
    correct_sum = correct_sum + logits.argmax(1).eq(y).sum().to(torch::kInt64);
  }
  const double examples = static_cast<double>(std::max<int64_t>(n, 1));
  return {loss_sum.to(torch::kCPU).item<double>() / examples,
          correct_sum.to(torch::kCPU).item<int64_t>() / examples};
}

std::string json_bool(bool value) {
  return value ? "true" : "false";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto process_begin = std::chrono::steady_clock::now();
    const Args args = parse_args(argc, argv);
    at::globalContext().setBenchmarkCuDNN(true);
    at::globalContext().setAllowTF32CuDNN(args.allow_tf32);
    at::globalContext().setAllowTF32CuBLAS(args.allow_tf32);
    torch::manual_seed(args.seed);
    const torch::Device device(args.device);
    if (device.is_cuda() && !torch::cuda::is_available()) {
      throw std::runtime_error("CUDA device requested but CUDA is not available");
    }

    const auto data_begin = std::chrono::steady_clock::now();
    auto train_data = load_cifar10_split(args.data_dir, "train");
    auto test_data = load_cifar10_split(args.data_dir, "test");
    train_data.images = train_data.images.to(device, /*non_blocking=*/false);
    train_data.labels = train_data.labels.to(device, /*non_blocking=*/false);
    test_data.images = test_data.images.to(device, /*non_blocking=*/false);
    test_data.labels = test_data.labels.to(device, /*non_blocking=*/false);
    sync_if_cuda(device);
    const double data_seconds = seconds_since(data_begin);

    const auto model_begin = std::chrono::steady_clock::now();
    ResNet152Cifar model;
    model->to(device);
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(args.lr)
                                    .momentum(args.momentum)
                                    .weight_decay(args.weight_decay));
    sync_if_cuda(device);
    const double model_init_seconds = seconds_since(model_begin);

    double warmup_seconds = 0.0;
    if (args.warmup_batches > 0) {
      const auto warmup_begin = std::chrono::steady_clock::now();
      (void)run_batches(model, optimizer, train_data.images, train_data.labels, args.batch_size, args.warmup_batches, true);
      sync_if_cuda(device);
      warmup_seconds = seconds_since(warmup_begin);
    }

    std::ofstream metrics;
    if (!args.metrics_csv.empty()) {
      metrics.open(args.metrics_csv);
      if (!metrics) {
        throw std::runtime_error("failed to open metrics csv: " + args.metrics_csv);
      }
      metrics << "backend,epoch,seconds,avg_loss,examples,batches,batch_size\n";
    }

    const int64_t examples = train_data.images.size(0);
    const int64_t test_examples = test_data.images.size(0);
    const int64_t batches = (examples + args.batch_size - 1) / args.batch_size;
    double epoch_train_seconds = 0.0;
    double final_loss = 0.0;
    for (int64_t epoch = 1; epoch <= args.epochs; ++epoch) {
      sync_if_cuda(device);
      const auto begin = std::chrono::steady_clock::now();
      final_loss = run_batches(model, optimizer, train_data.images, train_data.labels, args.batch_size, 0, true);
      sync_if_cuda(device);
      const auto end = std::chrono::steady_clock::now();
      const double seconds = std::chrono::duration<double>(end - begin).count();
      epoch_train_seconds += seconds;
      if (metrics) {
        metrics << "cpp_libtorch," << epoch << "," << std::setprecision(9) << seconds << ","
                << final_loss << "," << examples << "," << batches << "," << args.batch_size << "\n";
      }
      std::cout << "backend=cpp_libtorch epoch=" << epoch << " seconds=" << std::fixed << std::setprecision(6)
                << seconds << " avg_loss=" << final_loss << "\n";
    }

    const auto eval_begin = std::chrono::steady_clock::now();
    sync_if_cuda(device);
    const EvalMetrics train_metrics = evaluate(model, train_data.images, train_data.labels, args.batch_size);
    const EvalMetrics test_metrics = evaluate(model, test_data.images, test_data.labels, args.batch_size);
    sync_if_cuda(device);
    const double eval_seconds = seconds_since(eval_begin);
    const double end_to_end_seconds = seconds_since(process_begin);

    if (!args.summary_json.empty()) {
      std::ofstream out(args.summary_json);
      if (!out) {
        throw std::runtime_error("failed to open summary json: " + args.summary_json);
      }
      out << "{\n"
          << "  \"backend\": \"cpp_libtorch\",\n"
          << "  \"model\": \"resnet152_cifar\",\n"
          << "  \"dataset\": \"cifar10_train\",\n"
          << "  \"device\": \"" << args.device << "\",\n"
          << "  \"epochs\": " << args.epochs << ",\n"
          << "  \"batch_size\": " << args.batch_size << ",\n"
          << "  \"warmup_batches\": " << args.warmup_batches << ",\n"
          << "  \"lr\": " << args.lr << ",\n"
          << "  \"momentum\": " << args.momentum << ",\n"
          << "  \"weight_decay\": " << args.weight_decay << ",\n"
          << "  \"allow_tf32\": " << json_bool(args.allow_tf32) << ",\n"
          << "  \"examples\": " << examples << ",\n"
          << "  \"test_examples\": " << test_examples << ",\n"
          << "  \"batches_per_epoch\": " << batches << ",\n"
          << "  \"total_seconds\": " << std::setprecision(12) << end_to_end_seconds << ",\n"
          << "  \"end_to_end_seconds\": " << end_to_end_seconds << ",\n"
          << "  \"epoch_train_seconds\": " << epoch_train_seconds << ",\n"
          << "  \"seconds_per_epoch\": " << (epoch_train_seconds / static_cast<double>(args.epochs)) << ",\n"
          << "  \"data_seconds\": " << data_seconds << ",\n"
          << "  \"model_init_seconds\": " << model_init_seconds << ",\n"
          << "  \"warmup_seconds\": " << warmup_seconds << ",\n"
          << "  \"eval_seconds\": " << eval_seconds << ",\n"
          << "  \"final_loss\": " << final_loss << ",\n"
          << "  \"train_loss\": " << train_metrics.loss << ",\n"
          << "  \"train_accuracy\": " << train_metrics.accuracy << ",\n"
          << "  \"test_loss\": " << test_metrics.loss << ",\n"
          << "  \"test_accuracy\": " << test_metrics.accuracy << "\n"
          << "}\n";
    }

    std::cout << "metrics backend=cpp_libtorch train_loss=" << std::fixed << std::setprecision(6)
              << train_metrics.loss << " train_acc=" << train_metrics.accuracy
              << " test_loss=" << test_metrics.loss << " test_acc=" << test_metrics.accuracy << "\n";
    std::cout << "stages backend=cpp_libtorch data_seconds=" << std::fixed << std::setprecision(6)
              << data_seconds << " model_init_seconds=" << model_init_seconds
              << " warmup_seconds=" << warmup_seconds << " epoch_train_seconds=" << epoch_train_seconds
              << " eval_seconds=" << eval_seconds << " end_to_end_seconds=" << end_to_end_seconds << "\n";
    std::cout << "summary backend=cpp_libtorch epochs=" << args.epochs << " total_seconds=" << std::fixed
              << std::setprecision(6) << end_to_end_seconds
              << " seconds_per_epoch=" << (epoch_train_seconds / static_cast<double>(args.epochs))
              << " final_loss=" << final_loss << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "resnet152_cifar10_cpp_bench failed: " << e.what() << "\n";
    return 1;
  }
}
