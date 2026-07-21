#include "hyperion_mlx.h"

_Static_assert(HYP_STATUS_OK == 0, "status ABI changed");
_Static_assert(sizeof(HypCanaryInfo) == 224, "canary ABI layout changed");

int hyperion_abi_c_header_test(void) {
    HypCanaryInfo info = {0};
    return (int)info.abi_version;
}
