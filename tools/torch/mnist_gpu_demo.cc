#include <torch/torch.h>
#include <torch/cuda.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string data_dir = "data/mnist/raw";
  std::string device = "cuda:0";
  std::string metrics_csv;
  int64_t epochs = 1;
  int64_t batch_size = 256;
  int64_t max_train_batches = 0;
  int64_t max_test_batches = 0;
  double lr = 1.0e-3;
  int64_t seed = 17;
  bool stop_when_train_acc_gt_test = false;
};

std::string require_value(int argc, char** argv, int& i) {
  if (i + 1 >= argc) {
    throw std::invalid_argument(std::string("missing value for ") + argv[i]);
  }
  return argv[++i];
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--data-dir") {
      args.data_dir = require_value(argc, argv, i);
    } else if (flag == "--device") {
      args.device = require_value(argc, argv, i);
    } else if (flag == "--metrics-csv") {
      args.metrics_csv = require_value(argc, argv, i);
    } else if (flag == "--epochs") {
      args.epochs = std::stoll(require_value(argc, argv, i));
    } else if (flag == "--batch-size") {
      args.batch_size = std::stoll(require_value(argc, argv, i));
    } else if (flag == "--max-train-batches") {
      args.max_train_batches = std::stoll(require_value(argc, argv, i));
    } else if (flag == "--max-test-batches") {
      args.max_test_batches = std::stoll(require_value(argc, argv, i));
    } else if (flag == "--lr") {
      args.lr = std::stod(require_value(argc, argv, i));
    } else if (flag == "--seed") {
      args.seed = std::stoll(require_value(argc, argv, i));
    } else if (flag == "--stop-when-train-acc-gt-test") {
      args.stop_when_train_acc_gt_test = true;
    } else if (flag == "--help") {
      std::cout
          << "Usage: mnist_gpu_demo --data-dir data/mnist/raw [options]\n"
          << "Options:\n"
          << "  --device cuda:0\n"
          << "  --epochs 1\n"
          << "  --batch-size 256\n"
          << "  --max-train-batches 0   0 means full epoch\n"
          << "  --max-test-batches 0    0 means full test split\n"
          << "  --lr 1e-3\n"
          << "  --stop-when-train-acc-gt-test\n"
          << "  --metrics-csv path.csv\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }
  if (args.epochs <= 0 || args.batch_size <= 0) {
    throw std::invalid_argument("epochs and batch-size must be positive");
  }
  return args;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  const std::streamsize size = input.tellg();
  if (size <= 0) {
    throw std::runtime_error("empty file: " + path);
  }
  std::vector<uint8_t> data(static_cast<size_t>(size));
  input.seekg(0, std::ios::beg);
  if (!input.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("failed to read " + path);
  }
  return data;
}

int32_t read_be_i32(const std::vector<uint8_t>& data, size_t offset) {
  if (offset + 4 > data.size()) {
    throw std::runtime_error("IDX header truncated");
  }
  return (static_cast<int32_t>(data[offset]) << 24) |
         (static_cast<int32_t>(data[offset + 1]) << 16) |
         (static_cast<int32_t>(data[offset + 2]) << 8) |
         static_cast<int32_t>(data[offset + 3]);
}

torch::Tensor load_idx_images(const std::string& path) {
  std::vector<uint8_t> data = read_binary_file(path);
  const int32_t magic = read_be_i32(data, 0);
  const int32_t count = read_be_i32(data, 4);
  const int32_t rows = read_be_i32(data, 8);
  const int32_t cols = read_be_i32(data, 12);
  if (magic != 2051 || count <= 0 || rows != 28 || cols != 28) {
    throw std::runtime_error("invalid MNIST image IDX file: " + path);
  }
  const size_t payload = static_cast<size_t>(count) * rows * cols;
  if (data.size() != 16 + payload) {
    throw std::runtime_error("unexpected image IDX size: " + path);
  }
  auto raw = torch::from_blob(
                 data.data() + 16,
                 {count, 1, rows, cols},
                 torch::TensorOptions().dtype(torch::kUInt8))
                 .clone();
  return raw.to(torch::kFloat32).div_(255.0);
}

torch::Tensor load_idx_labels(const std::string& path) {
  std::vector<uint8_t> data = read_binary_file(path);
  const int32_t magic = read_be_i32(data, 0);
  const int32_t count = read_be_i32(data, 4);
  if (magic != 2049 || count <= 0) {
    throw std::runtime_error("invalid MNIST label IDX file: " + path);
  }
  if (data.size() != 8 + static_cast<size_t>(count)) {
    throw std::runtime_error("unexpected label IDX size: " + path);
  }
  auto raw = torch::from_blob(data.data() + 8, {count}, torch::TensorOptions().dtype(torch::kUInt8)).clone();
  return raw.to(torch::kLong);
}

struct MnistSplit {
  torch::Tensor images;
  torch::Tensor labels;
};

MnistSplit load_split(const std::string& data_dir, bool train) {
  const std::string image_name = train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte";
  const std::string label_name = train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte";
  MnistSplit split;
  split.images = load_idx_images(data_dir + "/" + image_name).contiguous();
  split.labels = load_idx_labels(data_dir + "/" + label_name).contiguous();
  if (split.images.size(0) != split.labels.size(0)) {
    throw std::runtime_error("image/label count mismatch in " + data_dir);
  }
  return split;
}

struct MnistNetImpl : torch::nn::Module {
  MnistNetImpl()
      : conv1(register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)))),
        conv2(register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)))),
        fc1(register_module("fc1", torch::nn::Linear(64 * 7 * 7, 128))),
        fc2(register_module("fc2", torch::nn::Linear(128, 10))) {}

  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(conv1->forward(x));
    x = torch::max_pool2d(x, 2);
    x = torch::relu(conv2->forward(x));
    x = torch::max_pool2d(x, 2);
    x = x.flatten(1);
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, 0.25, is_training());
    return fc2->forward(x);
  }

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
};

TORCH_MODULE(MnistNet);

struct EvalResult {
  double loss = 0.0;
  double accuracy = 0.0;
  int64_t examples = 0;
};

EvalResult evaluate(
    MnistNet& model,
    const MnistSplit& split,
    const torch::Device& device,
    int64_t batch_size,
    int64_t max_batches) {
  torch::NoGradGuard no_grad;
  model->eval();
  double loss_sum = 0.0;
  int64_t correct = 0;
  int64_t seen = 0;
  int64_t batches = 0;
  const int64_t total = split.images.size(0);
  for (int64_t begin = 0; begin < total; begin += batch_size) {
    if (max_batches > 0 && batches >= max_batches) {
      break;
    }
    const int64_t count = std::min(batch_size, total - begin);
    auto images = split.images.narrow(0, begin, count).to(device, /*non_blocking=*/true);
    auto labels = split.labels.narrow(0, begin, count).to(device, /*non_blocking=*/true);
    auto logits = model->forward(images);
    auto loss = torch::nn::functional::cross_entropy(logits, labels);
    loss_sum += loss.item<double>() * static_cast<double>(count);
    correct += logits.argmax(1).eq(labels).sum().item<int64_t>();
    seen += count;
    ++batches;
  }
  if (seen == 0) {
    throw std::runtime_error("evaluation saw zero examples");
  }
  return EvalResult{loss_sum / static_cast<double>(seen), static_cast<double>(correct) / seen, seen};
}

class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path) {
    if (!path.empty()) {
      output_.open(path);
      if (!output_) {
        throw std::runtime_error("failed to open metrics csv: " + path);
      }
      output_
          << "step,epoch,batch,train_loss,train_accuracy,train_eval_loss,train_eval_accuracy,test_loss,test_accuracy\n";
    }
  }

  void write(int64_t step,
             int64_t epoch,
             int64_t batch,
             double train_loss,
             double train_accuracy,
             double train_eval_loss,
             double train_eval_accuracy,
             double test_loss,
             double test_accuracy) {
    if (!output_) {
      return;
    }
    output_ << step << ',' << epoch << ',' << batch << ',' << std::setprecision(8) << train_loss << ','
            << train_accuracy << ',' << train_eval_loss << ',' << train_eval_accuracy << ',' << test_loss << ','
            << test_accuracy << '\n';
  }

 private:
  std::ofstream output_;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    torch::manual_seed(args.seed);
    if (args.device.rfind("cuda", 0) == 0 && !torch::cuda::is_available()) {
      throw std::runtime_error("CUDA device requested but torch::cuda::is_available() is false");
    }
    torch::Device device(args.device);

    MnistSplit train = load_split(args.data_dir, true);
    MnistSplit test = load_split(args.data_dir, false);
    MnistNet model;
    model->to(device);

    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(args.lr));
    CsvWriter csv(args.metrics_csv);

    std::cout << "mnist_gpu_demo device=" << args.device << " train_examples=" << train.images.size(0)
              << " test_examples=" << test.images.size(0) << " epochs=" << args.epochs
              << " batch_size=" << args.batch_size << " lr=" << args.lr << "\n";

    int64_t global_step = 0;
    double first_loss = -1.0;
    double last_loss = -1.0;
    for (int64_t epoch = 1; epoch <= args.epochs; ++epoch) {
      model->train();
      const auto order = torch::randperm(train.images.size(0), torch::TensorOptions().dtype(torch::kLong));
      double epoch_loss_sum = 0.0;
      int64_t epoch_correct = 0;
      int64_t epoch_seen = 0;
      int64_t batch_id = 0;
      for (int64_t begin = 0; begin < train.images.size(0); begin += args.batch_size) {
        if (args.max_train_batches > 0 && batch_id >= args.max_train_batches) {
          break;
        }
        const int64_t count = std::min(args.batch_size, train.images.size(0) - begin);
        auto index = order.narrow(0, begin, count);
        auto images = train.images.index_select(0, index).to(device, /*non_blocking=*/true);
        auto labels = train.labels.index_select(0, index).to(device, /*non_blocking=*/true);

        optimizer.zero_grad();
        auto logits = model->forward(images);
        auto loss = torch::nn::functional::cross_entropy(logits, labels);
        loss.backward();
        optimizer.step();

        const double loss_value = loss.item<double>();
        const int64_t correct = logits.argmax(1).eq(labels).sum().item<int64_t>();
        if (first_loss < 0.0) {
          first_loss = loss_value;
        }
        last_loss = loss_value;
        epoch_loss_sum += loss_value * static_cast<double>(count);
        epoch_correct += correct;
        epoch_seen += count;
        ++global_step;
        ++batch_id;

        const double train_acc = static_cast<double>(correct) / count;
        csv.write(global_step, epoch, batch_id, loss_value, train_acc, -1.0, -1.0, -1.0, -1.0);
        if (batch_id == 1 || batch_id % 25 == 0) {
          std::cout << "step=" << global_step << " epoch=" << epoch << " batch=" << batch_id
                    << " train_loss=" << std::fixed << std::setprecision(6) << loss_value
                    << " train_acc=" << train_acc << "\n";
        }
      }
      if (epoch_seen == 0) {
        throw std::runtime_error("training saw zero examples");
      }
      const EvalResult train_eval = evaluate(model, train, device, args.batch_size, args.max_test_batches);
      const EvalResult eval = evaluate(model, test, device, args.batch_size, args.max_test_batches);
      const double epoch_loss = epoch_loss_sum / static_cast<double>(epoch_seen);
      const double epoch_acc = static_cast<double>(epoch_correct) / epoch_seen;
      csv.write(global_step, epoch, -1, epoch_loss, epoch_acc, train_eval.loss, train_eval.accuracy, eval.loss, eval.accuracy);
      std::cout << "epoch=" << epoch << " train_loss=" << std::fixed << std::setprecision(6) << epoch_loss
                << " train_online_acc=" << epoch_acc << " train_eval_loss=" << train_eval.loss
                << " train_eval_acc=" << train_eval.accuracy << " test_loss=" << eval.loss
                << " test_acc=" << eval.accuracy << " test_examples=" << eval.examples << "\n";
      if (args.stop_when_train_acc_gt_test && train_eval.accuracy > eval.accuracy) {
        std::cout << "stop_reason=train_eval_acc_gt_test_acc"
                  << " epoch=" << epoch << " train_eval_acc=" << train_eval.accuracy
                  << " test_acc=" << eval.accuracy << "\n";
        break;
      }
    }

    std::cout << "mnist_gpu_demo completed first_batch_loss=" << std::fixed << std::setprecision(6)
              << first_loss << " last_batch_loss=" << last_loss
              << " loss_delta=" << (last_loss - first_loss) << "\n";
    std::cout << "metrics_csv=" << args.metrics_csv << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mnist_gpu_demo failed: " << e.what() << "\n";
    return 1;
  }
}
