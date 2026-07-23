cmake_minimum_required(VERSION 3.25)

project(hyperion_mlx LANGUAGES C CXX OBJCXX)

set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING "Hyperion macOS floor" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)

if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    message(FATAL_ERROR "Hyperion supports Apple Silicon only")
endif()

find_package(MLX 0.32.0 EXACT CONFIG REQUIRED)
if(NOT MLX_BUILD_METAL)
    message(FATAL_ERROR "Hyperion requires MLX 0.32.0 with Metal enabled")
endif()
if(NOT TARGET mlx)
    message(FATAL_ERROR "MLX 0.32.0 package did not define the mlx target")
endif()

mlx_build_metallib(
    TARGET hyperion_canary_metallib
    TITLE hyperion_canary
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/metal/canary.metal"
)

add_library(hyperion_mlx STATIC
    src/platform_policy.cc
    src/governor.cc
    src/runtime.mm
    src/abi_error.cc
    src/geometry.cc
    src/dispatch.cc
    src/kv.cc
    src/kv_cache.cc
    src/weights_loader.cc
    src/rope.cc
    src/forward.cc
    src/model.cc
)
add_dependencies(hyperion_mlx hyperion_canary_metallib)
target_include_directories(hyperion_mlx
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(hyperion_mlx PRIVATE mlx "-framework Foundation" "-framework Metal")
target_compile_definitions(hyperion_mlx PRIVATE
    HYPERION_METALLIB_DEFAULT_PATH="${CMAKE_CURRENT_BINARY_DIR}/hyperion_canary.metallib"
)
target_compile_options(hyperion_mlx PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
)

include(CTest)
if(BUILD_TESTING)
    add_library(hyperion_abi_c_header OBJECT tests/abi_c_header_test.c)
    target_include_directories(hyperion_abi_c_header PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_options(hyperion_abi_c_header PRIVATE -Wall -Wextra -Wpedantic -Werror)

    add_executable(hyperion_platform_policy_test
        tests/platform_policy_test.cc
        src/platform_policy.cc
    )
    target_include_directories(hyperion_platform_policy_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_platform_policy_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_platform_policy COMMAND hyperion_platform_policy_test)
    add_test(
        NAME hyperion_platform_policy_negative_control
        COMMAND hyperion_platform_policy_test --negative-control
    )
    set_tests_properties(hyperion_platform_policy_negative_control PROPERTIES WILL_FAIL TRUE)

    add_executable(hyperion_abi_contract_test tests/abi_contract_test.cc)
    target_link_libraries(hyperion_abi_contract_test PRIVATE hyperion_mlx)
    target_compile_options(hyperion_abi_contract_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_abi_contract COMMAND hyperion_abi_contract_test)

    add_executable(hyperion_runtime_canary_test tests/runtime_canary_test.cc)
    target_link_libraries(hyperion_runtime_canary_test PRIVATE hyperion_mlx)
    target_compile_options(hyperion_runtime_canary_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_runtime_canary COMMAND hyperion_runtime_canary_test)

    add_executable(hyperion_geometry_test tests/geometry_test.cc)
    target_link_libraries(hyperion_geometry_test PRIVATE hyperion_mlx)
    target_include_directories(hyperion_geometry_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_geometry_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_geometry COMMAND hyperion_geometry_test)

    add_executable(hyperion_dispatch_test tests/dispatch_test.cc)
    target_link_libraries(hyperion_dispatch_test PRIVATE hyperion_mlx)
    target_include_directories(hyperion_dispatch_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_dispatch_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_dispatch COMMAND hyperion_dispatch_test)

    add_executable(hyperion_kv_test tests/kv_test.cc)
    target_link_libraries(hyperion_kv_test PRIVATE hyperion_mlx)
    target_include_directories(hyperion_kv_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_kv_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_kv COMMAND hyperion_kv_test)

    # M5-gated: exercises the real MLX runtime (mx::array + slice_update + GPU
    # eval). Builds wherever MLX/Metal link; runs only on the self-hosted M5
    # (NOT in scripts/test-native.sh --model-free, like hyperion_runtime_canary).
    add_executable(hyperion_kv_cache_test tests/kv_cache_test.cc)
    target_link_libraries(hyperion_kv_cache_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_kv_cache_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_kv_cache_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_kv_cache COMMAND hyperion_kv_cache_test)

    # M5-gated: loads the REAL 12B q_proj and checks quantized_matmul vs a
    # dequantize+matmul reference. Builds wherever MLX/Metal link; runs only on
    # the self-hosted M5 (NOT in scripts/test-native.sh --model-free). Also
    # self-skips (exit 0) when HYPERION_12B_ARTIFACT is unset (CI).
    add_executable(hyperion_weights_test tests/weights_test.cc)
    target_link_libraries(hyperion_weights_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_weights_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_weights_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_weights COMMAND hyperion_weights_test)

    # M5-gated: the native forward-pass math (M2-2.3a) on a committed tiny
    # fixture, compared op-for-op against an inline MLX reference. Builds
    # wherever MLX/Metal link; runs only on the self-hosted M5 (NOT in
    # scripts/test-native.sh --model-free). Self-skips (exit 0) when the tiny
    # fixture dir is unset/absent (CI).
    add_executable(hyperion_forward_test tests/forward_test.cc)
    target_link_libraries(hyperion_forward_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_forward_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_forward_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_forward COMMAND hyperion_forward_test)

    # M5-gated: real-12B oracle parity (M2-2.3b). Loads the REAL 6.7 GB 12B via the
    # native ForwardPass and compares the post-final-norm hidden state to the committed
    # mlx-lm golden (gen_12b_hidden_golden.py). Builds wherever MLX/Metal link; runs only
    # on the self-hosted M5. Self-skips (exit 0) when HYPERION_12B_ARTIFACT is unset (CI).
    add_executable(hyperion_forward_12b_test tests/forward_12b_test.cc)
    target_link_libraries(hyperion_forward_12b_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_forward_12b_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_forward_12b_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_forward_12b COMMAND hyperion_forward_12b_test)

    # M5-gated: real-12B G1 token-exact seal (M2-2.7). Loads the REAL 12B via the native
    # ForwardPass, runs the chat-templated prompt through prefill + decode (offset>0
    # cached-prefix read) + the tied lm_head/softcap/greedy epilogue, and asserts greedy
    # tokens EXACT vs the committed mlx-lm oracle golden (gen_12b_greedy_golden.py).
    # Builds wherever MLX/Metal link; runs only on the self-hosted M5. Self-skips (exit 0)
    # when HYPERION_12B_ARTIFACT is unset (CI) or the golden is absent.
    add_executable(hyperion_forward_12b_decode_test tests/forward_12b_decode_test.cc)
    target_link_libraries(hyperion_forward_12b_decode_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_forward_12b_decode_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_forward_12b_decode_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_forward_12b_decode COMMAND hyperion_forward_12b_decode_test)

    # M5-gated: real-12B long-context token-exact seal (M2-2.6a). Loads the REAL 12B, runs
    # a 2436-token chat-templated prompt through chunked prefill (2048, crosses one chunk
    # boundary) + the sliding-ring rotation read past the 1024 window + decode, asserts
    # greedy tokens EXACT vs the committed mlx-lm oracle golden (gen_12b_long_golden.py).
    # Reports peak MLX bytes (the 2.6b governor gate input; not gated here). Builds where
    # MLX/Metal link; runs only on the self-hosted M5. Self-skips (exit 0) when
    # HYPERION_12B_ARTIFACT is unset (CI) or the golden is absent.
    add_executable(hyperion_forward_12b_long_test tests/forward_12b_long_test.cc)
    target_link_libraries(hyperion_forward_12b_long_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_forward_12b_long_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_forward_12b_long_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_forward_12b_long COMMAND hyperion_forward_12b_long_test)

    # M5-gated: the M2-2.6b governor predictive admission (model-free tier: builds
    # anywhere, runs only on the M5 with a live MLX memory counter). Self-skips
    # (exit 0) when no GPU is available (CI).
    add_executable(hyperion_governor_test tests/governor_test.cc)
    target_link_libraries(hyperion_governor_test PRIVATE hyperion_mlx mlx)
    target_include_directories(hyperion_governor_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_options(hyperion_governor_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME hyperion_governor COMMAND hyperion_governor_test)
endif()

install(TARGETS hyperion_mlx ARCHIVE DESTINATION lib)
install(FILES include/hyperion_mlx.h DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hyperion_canary.metallib" DESTINATION bin)
