#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CVERL_OK = 0,
  CVERL_ERR_INVALID_ARGUMENT = 1,
  CVERL_ERR_UNSUPPORTED = 2,
} cverl_status_t;

const char* cverl_status_string(cverl_status_t status);

#ifdef __cplusplus
}
#endif
