#include "cverl/distributed/nccl_collectives.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <utility>
#include <unistd.h>

namespace cverl::distributed {
namespace {

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void check_nccl(ncclResult_t status, const char* what) {
  if (status != ncclSuccess) {
    throw std::runtime_error(std::string(what) + ": " + ncclGetErrorString(status));
  }
}

ncclRedOp_t nccl_reduce_op(ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::Mean:
      return ncclSum;
    case ReduceOp::Max:
      return ncclMax;
  }
  throw std::invalid_argument("unsupported reduce op");
}

ncclDataType_t nccl_dtype(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return ncclFloat32;
  }
  if (dtype == torch::kFloat16) {
    return ncclFloat16;
  }
  if (dtype == torch::kBFloat16) {
    return ncclBfloat16;
  }
  if (dtype == torch::kInt32) {
    return ncclInt32;
  }
  if (dtype == torch::kInt64) {
    return ncclInt64;
  }
  throw std::invalid_argument("unsupported NCCL tensor dtype");
}

void require_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  if (!tensor.is_cuda() || !tensor.is_contiguous()) {
    throw std::invalid_argument(std::string(name) + " must be a contiguous CUDA tensor");
  }
}

void require_all_rank_group(const std::vector<int64_t>& group, int64_t world_size) {
  if (static_cast<int64_t>(group.size()) != world_size) {
    throw std::invalid_argument("current NCCL implementation expects a full-world group");
  }
  for (int64_t i = 0; i < world_size; ++i) {
    if (group[static_cast<size_t>(i)] != i) {
      throw std::invalid_argument("current NCCL implementation expects group [0..world_size)");
    }
  }
}

void wait_current_stream_before_nccl(cudaStream_t nccl_stream, int device_index) {
  cudaEvent_t event = nullptr;
  check_cuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreateWithFlags");
  auto current_stream = at::cuda::getCurrentCUDAStream(device_index).stream();
  check_cuda(cudaEventRecord(event, current_stream), "cudaEventRecord before NCCL");
  check_cuda(cudaStreamWaitEvent(nccl_stream, event, 0), "cudaStreamWaitEvent before NCCL");
  check_cuda(cudaEventDestroy(event), "cudaEventDestroy before NCCL");
}

ncclUniqueId decode_unique_id(const NcclUniqueIdBytes& bytes) {
  if (bytes.bytes.size() != NCCL_UNIQUE_ID_BYTES) {
    throw std::invalid_argument("invalid NCCL unique id byte length");
  }
  ncclUniqueId id;
  std::memcpy(&id, bytes.bytes.data(), NCCL_UNIQUE_ID_BYTES);
  return id;
}

}  // namespace

NcclUniqueIdBytes create_nccl_unique_id() {
  ncclUniqueId id;
  check_nccl(ncclGetUniqueId(&id), "ncclGetUniqueId");
  NcclUniqueIdBytes out;
  out.bytes.resize(NCCL_UNIQUE_ID_BYTES);
  std::memcpy(out.bytes.data(), &id, NCCL_UNIQUE_ID_BYTES);
  return out;
}

void write_nccl_unique_id_file(const std::string& path, const NcclUniqueIdBytes& id) {
  const std::string tmp_path = path + ".tmp." + std::to_string(::getpid());
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open NCCL id file for write: " + tmp_path);
  }
  out.write(id.bytes.data(), static_cast<std::streamsize>(id.bytes.size()));
  if (!out) {
    throw std::runtime_error("failed to write NCCL id file: " + tmp_path);
  }
  out.close();
  std::filesystem::rename(tmp_path, path);
}

NcclUniqueIdBytes read_nccl_unique_id_file(const std::string& path) {
  for (int i = 0; i < 200; ++i) {
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) == NCCL_UNIQUE_ID_BYTES) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open NCCL id file for read: " + path);
  }
  NcclUniqueIdBytes out;
  out.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  if (out.bytes.size() != NCCL_UNIQUE_ID_BYTES) {
    throw std::runtime_error("invalid NCCL id file size: " + path);
  }
  return out;
}

NcclCollectives::NcclCollectives(int64_t rank,
                                 int64_t world_size,
                                 int device_index,
                                 const NcclUniqueIdBytes& unique_id,
                                 bool synchronize_after_collective)
    : rank_(rank),
      world_size_(world_size),
      device_index_(device_index),
      synchronize_after_collective_(synchronize_after_collective) {
  if (rank < 0 || rank >= world_size || world_size <= 0) {
    throw std::invalid_argument("invalid NCCL rank/world_size");
  }
  check_cuda(cudaSetDevice(device_index_), "cudaSetDevice");
  check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  ncclUniqueId id = decode_unique_id(unique_id);
  check_nccl(ncclCommInitRank(&comm_, static_cast<int>(world_size_), id, static_cast<int>(rank_)), "ncclCommInitRank");
}

NcclCollectives::~NcclCollectives() {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(stream_);
  }
  for (auto& op : pending_ops_) {
    if (op.done != nullptr) {
      (void)cudaEventDestroy(op.done);
    }
  }
  pending_ops_.clear();
  if (comm_ != nullptr) {
    (void)ncclCommDestroy(comm_);
  }
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(stream_);
  }
}

void NcclCollectives::synchronize() {
  if (stream_ != nullptr) {
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }
  for (auto& op : pending_ops_) {
    if (op.done != nullptr) {
      check_cuda(cudaEventDestroy(op.done), "cudaEventDestroy pending NCCL op");
    }
  }
  pending_ops_.clear();
}

void NcclCollectives::finish_nccl_op(std::vector<torch::Tensor> keep_alive) {
  if (synchronize_after_collective_) {
    synchronize();
    return;
  }

  collect_finished_pending_ops();
  cudaEvent_t done = nullptr;
  check_cuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreateWithFlags pending NCCL op");
  check_cuda(cudaEventRecord(done, stream_), "cudaEventRecord pending NCCL op");
  pending_ops_.push_back(PendingOp{done, std::move(keep_alive)});

  auto current_stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
  check_cuda(cudaStreamWaitEvent(current_stream, done, 0), "cudaStreamWaitEvent pending NCCL op");
}

void NcclCollectives::collect_finished_pending_ops() {
  size_t write = 0;
  for (size_t read = 0; read < pending_ops_.size(); ++read) {
    auto& op = pending_ops_[read];
    const auto status = cudaEventQuery(op.done);
    if (status == cudaSuccess) {
      check_cuda(cudaEventDestroy(op.done), "cudaEventDestroy pending NCCL op");
      continue;
    }
    if (status != cudaErrorNotReady) {
      check_cuda(status, "cudaEventQuery pending NCCL op");
    }
    if (write != read) {
      pending_ops_[write] = std::move(op);
    }
    ++write;
  }
  pending_ops_.resize(write);
}

void NcclCollectives::barrier() {
  auto token = torch::ones({1}, torch::TensorOptions().device(torch::kCUDA, device_index_).dtype(torch::kFloat32));
  std::vector<int64_t> group;
  group.reserve(static_cast<size_t>(world_size_));
  for (int64_t i = 0; i < world_size_; ++i) {
    group.push_back(i);
  }
  (void)all_reduce(token, ReduceOp::Sum, group);
  synchronize();
}

torch::Tensor NcclCollectives::broadcast(const torch::Tensor& input,
                                         int64_t root,
                                         const std::vector<int64_t>& group) {
  require_all_rank_group(group, world_size_);
  require_cuda_contiguous(input, "broadcast input");
  if (root < 0 || root >= world_size_) {
    throw std::invalid_argument("broadcast root out of range");
  }
  auto out = torch::empty_like(input);
  void* send_recv = rank_ == root ? input.data_ptr() : out.data_ptr();
  if (rank_ == root) {
    wait_current_stream_before_nccl(stream_, device_index_);
  }
  check_nccl(ncclBroadcast(send_recv, out.data_ptr(), input.numel(), nccl_dtype(input.scalar_type()),
                           static_cast<int>(root), comm_, stream_),
             "ncclBroadcast");
  finish_nccl_op({input, out});
  if (rank_ == root) {
    out.copy_(input);
  }
  return out;
}

torch::Tensor NcclCollectives::all_reduce(const torch::Tensor& input, ReduceOp op, const std::vector<int64_t>& group) {
  require_all_rank_group(group, world_size_);
  require_cuda_contiguous(input, "all_reduce input");
  auto out = torch::empty_like(input);
  wait_current_stream_before_nccl(stream_, device_index_);
  check_nccl(ncclAllReduce(input.data_ptr(), out.data_ptr(), input.numel(), nccl_dtype(input.scalar_type()),
                           nccl_reduce_op(op), comm_, stream_),
             "ncclAllReduce");
  finish_nccl_op({input, out});
  if (op == ReduceOp::Mean) {
    out.div_(static_cast<double>(group.size()));
  }
  return out;
}

torch::Tensor NcclCollectives::all_gather(const torch::Tensor& input, const std::vector<int64_t>& group, int64_t dim) {
  require_all_rank_group(group, world_size_);
  require_cuda_contiguous(input, "all_gather input");
  if (dim != 0) {
    throw std::invalid_argument("NCCL all_gather currently supports dim=0 only");
  }
  std::vector<int64_t> shape = input.sizes().vec();
  shape[0] *= static_cast<int64_t>(group.size());
  auto out = torch::empty(shape, input.options());
  wait_current_stream_before_nccl(stream_, device_index_);
  check_nccl(ncclAllGather(input.data_ptr(), out.data_ptr(), input.numel(), nccl_dtype(input.scalar_type()), comm_, stream_),
             "ncclAllGather");
  finish_nccl_op({input, out});
  return out;
}

torch::Tensor NcclCollectives::reduce_scatter(const torch::Tensor& input,
                                              ReduceOp op,
                                              const std::vector<int64_t>& group,
                                              int64_t dim) {
  require_all_rank_group(group, world_size_);
  require_cuda_contiguous(input, "reduce_scatter input");
  if (dim != 0 || input.size(0) % static_cast<int64_t>(group.size()) != 0) {
    throw std::invalid_argument("NCCL reduce_scatter currently requires dim=0 evenly divisible by group size");
  }
  std::vector<int64_t> shape = input.sizes().vec();
  shape[0] /= static_cast<int64_t>(group.size());
  auto out = torch::empty(shape, input.options());
  wait_current_stream_before_nccl(stream_, device_index_);
  check_nccl(ncclReduceScatter(input.data_ptr(), out.data_ptr(), out.numel(), nccl_dtype(input.scalar_type()),
                               nccl_reduce_op(op), comm_, stream_),
             "ncclReduceScatter");
  finish_nccl_op({input, out});
  if (op == ReduceOp::Mean) {
    out.div_(static_cast<double>(group.size()));
  }
  return out;
}

void NcclCollectives::send(const torch::Tensor& input, int64_t peer) {
  require_cuda_contiguous(input, "send input");
  wait_current_stream_before_nccl(stream_, device_index_);
  check_nccl(ncclSend(input.data_ptr(), input.numel(), nccl_dtype(input.scalar_type()), static_cast<int>(peer), comm_, stream_),
             "ncclSend");
  finish_nccl_op({input});
}

torch::Tensor NcclCollectives::recv_like(const torch::Tensor& like, int64_t peer) {
  require_cuda_contiguous(like, "recv_like tensor");
  auto out = torch::empty_like(like);
  check_nccl(ncclRecv(out.data_ptr(), out.numel(), nccl_dtype(out.scalar_type()), static_cast<int>(peer), comm_, stream_),
             "ncclRecv");
  finish_nccl_op({out});
  return out;
}

}  // namespace cverl::distributed
