#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CVERL_DTYPE_F32 = 0,
} cverl_dtype_t;

typedef enum {
  CVERL_DEVICE_CPU = 0,
  CVERL_DEVICE_CUDA = 1,
} cverl_device_t;

typedef struct {
  void* data;
  cverl_dtype_t dtype;
  cverl_device_t device;
  int64_t rows;
  int64_t cols;
} cverl_tensor2d_t;

typedef struct {
  const void* data;
  cverl_dtype_t dtype;
  cverl_device_t device;
  int64_t rows;
  int64_t cols;
} cverl_const_tensor2d_t;

#ifdef __cplusplus
}
#endif
