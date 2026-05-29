#include "cverl/status.h"

const char* cverl_status_string(cverl_status_t status) {
  switch (status) {
    case CVERL_OK:
      return "ok";
    case CVERL_ERR_INVALID_ARGUMENT:
      return "invalid argument";
    case CVERL_ERR_UNSUPPORTED:
      return "unsupported";
  }
  return "unknown";
}
