#include <metal_stdlib>

using namespace metal;

kernel void hyperion_canary_add_one(
    device float* value [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0) {
        value[0] += 1.0F;
    }
}
