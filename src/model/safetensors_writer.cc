#include "cverl/model/safetensors_writer.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace cverl {
namespace {

struct TensorEntry {
  std::string name;
  torch::Tensor tensor;
  std::string safetensors_dtype;
  uint64_t data_begin = 0;
  uint64_t data_end = 0;
};

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char ch : s) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(ch);
        break;
    }
  }
  return out;
}

torch::ScalarType target_scalar_type(const std::string& dtype) {
  if (dtype == "float32" || dtype == "f32" || dtype == "F32") {
    return torch::kFloat32;
  }
  if (dtype == "float16" || dtype == "f16" || dtype == "F16") {
    return torch::kFloat16;
  }
  if (dtype == "bfloat16" || dtype == "bf16" || dtype == "BF16") {
    return torch::kBFloat16;
  }
  throw std::invalid_argument("unsupported safetensors export dtype: " + dtype);
}

std::string safetensors_dtype(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return "F32";
  }
  if (dtype == torch::kFloat16) {
    return "F16";
  }
  if (dtype == torch::kBFloat16) {
    return "BF16";
  }
  if (dtype == torch::kInt64) {
    return "I64";
  }
  if (dtype == torch::kInt32) {
    return "I32";
  }
  if (dtype == torch::kUInt8) {
    return "U8";
  }
  throw std::invalid_argument("unsupported tensor dtype for safetensors export");
}

void write_u64_le(std::ofstream& out, uint64_t value) {
  unsigned char bytes[8];
  for (int i = 0; i < 8; ++i) {
    bytes[i] = static_cast<unsigned char>((value >> (8 * i)) & 0xff);
  }
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::string build_header(const std::vector<TensorEntry>& entries) {
  std::ostringstream oss;
  oss << "{\"__metadata__\":{\"format\":\"pt\"}";
  for (const auto& entry : entries) {
    oss << ",\"" << json_escape(entry.name) << "\":{\"dtype\":\""
        << entry.safetensors_dtype << "\",\"shape\":[";
    for (int64_t i = 0; i < entry.tensor.dim(); ++i) {
      if (i != 0) {
        oss << ",";
      }
      oss << entry.tensor.size(i);
    }
    oss << "],\"data_offsets\":[" << entry.data_begin << "," << entry.data_end << "]}";
  }
  oss << "}";
  std::string header = oss.str();
  while ((8 + header.size()) % 8 != 0) {
    header.push_back(' ');
  }
  return header;
}

}  // namespace

void write_safetensors(const std::string& path,
                       const std::unordered_map<std::string, torch::Tensor>& tensors,
                       const std::string& dtype) {
  if (tensors.empty()) {
    throw std::invalid_argument("write_safetensors requires at least one tensor");
  }
  std::vector<TensorEntry> entries;
  entries.reserve(tensors.size());
  const auto export_dtype = target_scalar_type(dtype);

  uint64_t offset = 0;
  for (const auto& kv : tensors) {
    if (!kv.second.defined()) {
      throw std::invalid_argument("undefined tensor for safetensors key: " + kv.first);
    }
    TensorEntry entry;
    entry.name = kv.first;
    entry.tensor = kv.second.detach().to(torch::kCPU).to(export_dtype).contiguous();
    entry.safetensors_dtype = safetensors_dtype(entry.tensor.scalar_type());
    entry.data_begin = offset;
    offset += static_cast<uint64_t>(entry.tensor.nbytes());
    entry.data_end = offset;
    entries.push_back(std::move(entry));
  }

  const std::string header = build_header(entries);
  auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open safetensors output: " + path);
  }
  write_u64_le(out, static_cast<uint64_t>(header.size()));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& entry : entries) {
    out.write(static_cast<const char*>(entry.tensor.data_ptr()),
              static_cast<std::streamsize>(entry.tensor.nbytes()));
  }
  if (!out) {
    throw std::runtime_error("failed to write safetensors output: " + path);
  }
}

}  // namespace cverl
