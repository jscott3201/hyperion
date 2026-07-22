//! Safe Rust wrapper over Hyperion's narrow native C ABI.
//!
//! This is the only workspace crate permitted to contain `unsafe` code.

use std::{
    error::Error as StdError,
    ffi::{CStr, CString},
    fmt, io,
    os::raw::{c_char, c_int},
};

mod raw {
    use super::{c_char, c_int};
    use std::ffi::c_void;

    // Opaque, magic-tagged handle types (pointers to never-instantiated enums
    // give type safety: a HypKvState cannot be passed where a HypModel is expected).
    pub enum HypModelOpaque {}
    pub enum HypKvStateOpaque {}
    pub enum HypStepResultOpaque {}
    pub type HypModel = *mut HypModelOpaque;
    pub type HypKvState = *mut HypKvStateOpaque;
    pub type HypStepResult = *mut HypStepResultOpaque;

    pub const HYP_STATUS_OK: c_int = 0;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct HypCanaryInfo {
        pub abi_version: u32,
        pub mlx_compile_major: u32,
        pub mlx_compile_minor: u32,
        pub mlx_compile_patch: u32,
        pub macos_major: u32,
        pub macos_minor: u32,
        pub macos_patch: u32,
        pub gpu_family: u32,
        pub recommended_working_set_bytes: u64,
        pub effective_budget_bytes: u64,
        pub soft_watermark_bytes: u64,
        pub mlx_probe_value: f32,
        pub metallib_probe_value: f32,
        pub mlx_runtime_version: [c_char; 32],
        pub gpu_name: [c_char; 128],
    }

    /// Darwin `rusage_info_v4` from `<sys/resource.h>`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct RusageInfoV4 {
        pub uuid: [u8; 16],
        pub user_time: u64,
        pub system_time: u64,
        pub pkg_idle_wkups: u64,
        pub interrupt_wkups: u64,
        pub pageins: u64,
        pub wired_size: u64,
        pub resident_size: u64,
        pub phys_footprint: u64,
        pub proc_start_abstime: u64,
        pub proc_exit_abstime: u64,
        pub child_user_time: u64,
        pub child_system_time: u64,
        pub child_pkg_idle_wkups: u64,
        pub child_interrupt_wkups: u64,
        pub child_pageins: u64,
        pub child_elapsed_abstime: u64,
        pub diskio_bytesread: u64,
        pub diskio_byteswritten: u64,
        pub cpu_time_qos_default: u64,
        pub cpu_time_qos_maintenance: u64,
        pub cpu_time_qos_background: u64,
        pub cpu_time_qos_utility: u64,
        pub cpu_time_qos_legacy: u64,
        pub cpu_time_qos_user_initiated: u64,
        pub cpu_time_qos_user_interactive: u64,
        pub billed_system_time: u64,
        pub serviced_system_time: u64,
        pub logical_writes: u64,
        pub lifetime_max_phys_footprint: u64,
        pub instructions: u64,
        pub cycles: u64,
        pub billed_energy: u64,
        pub serviced_energy: u64,
        pub interval_max_phys_footprint: u64,
        pub runnable_time: u64,
    }

    pub const RUSAGE_INFO_V4: c_int = 4;

    impl Default for HypCanaryInfo {
        fn default() -> Self {
            Self {
                abi_version: 0,
                mlx_compile_major: 0,
                mlx_compile_minor: 0,
                mlx_compile_patch: 0,
                macos_major: 0,
                macos_minor: 0,
                macos_patch: 0,
                gpu_family: 0,
                recommended_working_set_bytes: 0,
                effective_budget_bytes: 0,
                soft_watermark_bytes: 0,
                mlx_probe_value: 0.0,
                metallib_probe_value: 0.0,
                mlx_runtime_version: [0; 32],
                gpu_name: [0; 128],
            }
        }
    }

    unsafe extern "C" {
        pub fn hyp_runtime_canary(out_info: *mut HypCanaryInfo) -> c_int;
        pub fn hyp_last_error(buffer: *mut c_char, buffer_len: usize) -> c_int;
    }

    #[link(name = "proc")]
    unsafe extern "C" {
        pub fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *mut c_void) -> c_int;
    }

    // M2 model + step ABI.

    /// C `HypTextModelType` enum value.
    pub const HYP_GEMMA4_TEXT: c_int = 0;
    pub const HYP_GEMMA4_UNIFIED_TEXT: c_int = 1;
    /// C `HypLayerType` enum value.
    pub const HYP_LAYER_SLIDING: c_int = 0;
    pub const HYP_LAYER_FULL: c_int = 1;
    /// C `HypGovernorState` enum value.
    pub const HYP_GOVERNOR_READY: c_int = 0;
    pub const HYP_GOVERNOR_SOFT_PAUSED: c_int = 1;
    pub const HYP_GOVERNOR_HARD_REJECT: c_int = 2;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct HypRopeSpec {
        pub theta: f64,
        pub has_partial_rotary_factor: c_int,
        pub partial_rotary_factor: f32,
        pub proportional: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct HypMoeConfig {
        pub num_experts: u32,
        pub top_k: u32,
        pub moe_intermediate_size: u32,
    }

    /// Validated geometry crossing the ABI (built from hyperion_model::Geometry at 2.2).
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct HypGeometryParams {
        pub model_type: c_int,
        pub hidden_size: u32,
        pub intermediate_size: u32,
        pub num_hidden_layers: u32,
        pub layer_types: *const c_int,
        pub num_attention_heads: u32,
        pub head_dim_local: u32,
        pub head_dim_global: u32,
        pub num_kv_heads_local: u32,
        pub num_kv_heads_global: u32,
        pub attention_k_eq_v_global: c_int,
        pub num_kv_shared_layers: u32,
        pub sliding_window: u32,
        pub rope_local: HypRopeSpec,
        pub rope_global: HypRopeSpec,
        pub final_logit_softcapping: f32,
        pub rms_norm_eps: f32,
        pub attention_bias: c_int,
        pub vocab_size: u32,
        pub max_position_embeddings: u32,
        pub tie_word_embeddings: c_int,
        pub ple_hidden_per_layer_input: u32,
        pub ple_vocab_per_layer_input: u32,
        pub use_double_wide_mlp: c_int,
        pub has_moe: c_int,
        pub moe: HypMoeConfig,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct HypTokenStream {
        pub tokens: *const u32,
        pub count: u32,
        pub is_prompt: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct HypStepResultFields {
        pub token_id: u32,
        pub logit: f32,
        pub peak_mlx_bytes: u64,
        pub active_mlx_bytes: u64,
        pub phys_footprint_bytes: u64,
        pub local_kv_bytes: u64,
        pub global_kv_bytes: u64,
        pub local_kv_eval_ms: f32,
        pub global_kv_eval_ms: f32,
        pub governor_state: c_int,
        pub near_tie_events: u32,
    }

    unsafe extern "C" {
        pub fn hyp_model_create(out_model: *mut HypModel) -> c_int;
        pub fn hyp_model_free(model: *mut HypModel) -> c_int;
        pub fn hyp_model_load(
            model: HypModel,
            geometry: *const HypGeometryParams,
            weights_path: *const c_char,
        ) -> c_int;
        pub fn hyp_kvstate_create(model: HypModel, out_kvstate: *mut HypKvState) -> c_int;
        pub fn hyp_kvstate_free(kvstate: *mut HypKvState) -> c_int;
        pub fn hyp_prefill_chunk(
            model: HypModel,
            kvstate: HypKvState,
            tokens: *const HypTokenStream,
            out_result: HypStepResult,
        ) -> c_int;
        pub fn hyp_decode_block(
            model: HypModel,
            kvstate: HypKvState,
            n_tokens: u32,
            out_result: HypStepResult,
        ) -> c_int;
        pub fn hyp_step_result_create(out_result: *mut HypStepResult) -> c_int;
        pub fn hyp_step_result_free(result: *mut HypStepResult) -> c_int;
    }
}

// The M2 ABI types + enum constants are the public FFI surface the server (M3)
// and bench (G1 harness) build and pass; re-export them from the crate root.
pub use raw::{
    HYP_GEMMA4_TEXT, HYP_GEMMA4_UNIFIED_TEXT, HYP_GOVERNOR_HARD_REJECT, HYP_GOVERNOR_READY,
    HYP_GOVERNOR_SOFT_PAUSED, HYP_LAYER_FULL, HYP_LAYER_SLIDING, HypGeometryParams, HypMoeConfig,
    HypRopeSpec, HypStepResultFields, HypTokenStream,
};

/// Stable native status taxonomy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Status {
    /// Operation succeeded.
    Ok,
    /// A caller argument was invalid.
    InvalidArgument,
    /// A required file or object was not found.
    NotFound,
    /// An I/O operation failed.
    Io,
    /// Predictive memory admission rejected the operation.
    OomGovernor,
    /// The platform or requested capability is unsupported.
    Unsupported,
    /// An internal invariant failed.
    Internal,
    /// The operation was cancelled.
    Cancelled,
    /// A native status newer than this Rust wrapper was returned.
    Unknown(i32),
}

impl From<c_int> for Status {
    fn from(value: c_int) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::InvalidArgument,
            2 => Self::NotFound,
            3 => Self::Io,
            4 => Self::OomGovernor,
            5 => Self::Unsupported,
            6 => Self::Internal,
            7 => Self::Cancelled,
            other => Self::Unknown(other),
        }
    }
}

/// Error reported by the native runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    /// Typed native status.
    pub status: Status,
    /// Thread-local native diagnostic.
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}: {}", self.status, self.message)
    }
}

impl StdError for Error {}

/// Successful M0 startup-canary evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CanaryInfo {
    /// Native ABI revision.
    pub abi_version: u32,
    /// MLX header version linked into the wrapper.
    pub mlx_compile_version: (u32, u32, u32),
    /// MLX dylib version loaded at runtime.
    pub mlx_runtime_version: String,
    /// Running macOS version.
    pub macos_version: (u32, u32, u32),
    /// Highest public Apple GPU family known to this build and supported by the device.
    pub gpu_family: u32,
    /// Metal's performance-safe working-set recommendation.
    pub recommended_working_set_bytes: u64,
    /// `min(12 GiB, floor(recommended * 0.949))`.
    pub effective_budget_bytes: u64,
    /// Ninety percent of the effective budget.
    pub soft_watermark_bytes: u64,
    /// Deterministic result of the real MLX tensor probe.
    pub mlx_probe_value: f32,
    /// Deterministic result of the custom metallib probe.
    pub metallib_probe_value: f32,
    /// Public Metal device name, for diagnostics only.
    pub gpu_name: String,
}

/// Process memory counters returned by Darwin's `proc_pid_rusage` API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessMemorySample {
    /// Cumulative page-in count, used as a pressure diagnostic.
    pub pageins: u64,
    /// Bytes currently wired for the process.
    pub wired_size_bytes: u64,
    /// Resident-set bytes currently attributed to the process.
    pub resident_size_bytes: u64,
    /// Current physical footprint, including compressed-memory accounting.
    pub phys_footprint_bytes: u64,
    /// Largest physical footprint over the process lifetime.
    pub lifetime_max_phys_footprint_bytes: u64,
    /// Largest physical footprint over the kernel's current accounting interval.
    pub interval_max_phys_footprint_bytes: u64,
}

/// Sample one live macOS process without exposing Darwin FFI outside this crate.
pub fn sample_process_memory(pid: u32) -> io::Result<ProcessMemorySample> {
    let pid = c_int::try_from(pid)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "PID exceeds c_int"))?;
    let mut usage = raw::RusageInfoV4::default();
    // SAFETY: `usage` is writable and has the exact `rusage_info_v4` C layout selected by the
    // flavor. `pid` is a checked positive-width Darwin process identifier.
    let result = unsafe {
        raw::proc_pid_rusage(
            pid,
            raw::RUSAGE_INFO_V4,
            (&raw mut usage).cast::<std::ffi::c_void>(),
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(ProcessMemorySample {
        pageins: usage.pageins,
        wired_size_bytes: usage.wired_size,
        resident_size_bytes: usage.resident_size,
        phys_footprint_bytes: usage.phys_footprint,
        lifetime_max_phys_footprint_bytes: usage.lifetime_max_phys_footprint,
        interval_max_phys_footprint_bytes: usage.interval_max_phys_footprint,
    })
}

/// Run the real stateless M0 platform/native canary.
pub fn runtime_canary() -> Result<CanaryInfo, Error> {
    let mut value = raw::HypCanaryInfo::default();
    // SAFETY: `value` is a writable instance of the exact repr(C) output type.
    let status = unsafe { raw::hyp_runtime_canary(&raw mut value) };
    if status != raw::HYP_STATUS_OK {
        return Err(native_error(status));
    }
    Ok(CanaryInfo {
        abi_version: value.abi_version,
        mlx_compile_version: (
            value.mlx_compile_major,
            value.mlx_compile_minor,
            value.mlx_compile_patch,
        ),
        mlx_runtime_version: char_array(&value.mlx_runtime_version),
        macos_version: (value.macos_major, value.macos_minor, value.macos_patch),
        gpu_family: value.gpu_family,
        recommended_working_set_bytes: value.recommended_working_set_bytes,
        effective_budget_bytes: value.effective_budget_bytes,
        soft_watermark_bytes: value.soft_watermark_bytes,
        mlx_probe_value: value.mlx_probe_value,
        metallib_probe_value: value.metallib_probe_value,
        gpu_name: char_array(&value.gpu_name),
    })
}

fn native_error(status: c_int) -> Error {
    let mut buffer = [0 as c_char; 1024];
    // SAFETY: the buffer is writable for exactly `buffer.len()` bytes.
    let error_status = unsafe { raw::hyp_last_error(buffer.as_mut_ptr(), buffer.len()) };
    let message = if error_status == raw::HYP_STATUS_OK {
        // SAFETY: the native function guarantees NUL termination for a non-empty buffer.
        unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    } else {
        "native error text was unavailable".to_owned()
    };
    Error {
        status: status.into(),
        message,
    }
}

fn char_array<const N: usize>(value: &[c_char; N]) -> String {
    // SAFETY: every native fixed string is initialized to zero and copied with a terminal NUL.
    unsafe { CStr::from_ptr(value.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

// --- M2 model + step ABI safe wrappers (RAII; magic-tagged handles) --------

/// Owned model handle. The native `hyp_model_free` nulls on drop, so double-drop
/// is a no-op. Not `Send` — the engine thread owns all native state (02-architecture).
pub struct Model(raw::HypModel);

impl Model {
    /// Allocate an empty magic-tagged model handle.
    pub fn create() -> Result<Self, Error> {
        let mut handle: raw::HypModel = std::ptr::null_mut();
        // SAFETY: `handle` is a writable out-pointer for the magic-tagged handle.
        let status = unsafe { raw::hyp_model_create(&mut handle) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(Self(handle))
    }

    /// Load weights + build the graph (stub -> `Unsupported` until M2-2.2).
    pub fn load(&self, geometry: &raw::HypGeometryParams, weights_path: &str) -> Result<(), Error> {
        let path = CString::new(weights_path).map_err(|_| Error {
            status: Status::InvalidArgument,
            message: "weights_path contains a NUL byte".into(),
        })?;
        // SAFETY: `self.0` is a valid handle; `geometry` outlives the call; `path` is NUL-terminated.
        let status = unsafe { raw::hyp_model_load(self.0, geometry, path.as_ptr()) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let mut handle = self.0;
        // SAFETY: owned handle; free nulls `handle`. Idempotent on a null handle.
        unsafe {
            let _ = raw::hyp_model_free(&mut handle);
        }
        self.0 = handle;
    }
}

/// Owned KV-state handle (bound to a model).
pub struct KvState(raw::HypKvState);

impl KvState {
    /// Allocate an empty magic-tagged KV-state bound to `model` (real KV at M2-2.1).
    pub fn create(model: &Model) -> Result<Self, Error> {
        let mut handle: raw::HypKvState = std::ptr::null_mut();
        // SAFETY: `model` is a valid handle; `handle` is a writable out-pointer.
        let status = unsafe { raw::hyp_kvstate_create(model.0, &mut handle) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(Self(handle))
    }

    /// Prefill a prompt chunk (stub -> `Unsupported` until M2-2.6).
    pub fn prefill_chunk(
        &self,
        model: &Model,
        tokens: &HypTokenStream,
        result: &StepResult,
    ) -> Result<(), Error> {
        // SAFETY: model/kvstate/result are valid handles; `tokens` outlives the call.
        let status = unsafe { raw::hyp_prefill_chunk(model.0, self.0, tokens, result.0) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(())
    }

    /// Decode `n_tokens` into the KV state (stub -> `Unsupported` until M2-2.6/2.7).
    pub fn decode_block(
        &self,
        model: &Model,
        n_tokens: u32,
        result: &StepResult,
    ) -> Result<(), Error> {
        // SAFETY: model/kvstate/result are valid handles.
        let status = unsafe { raw::hyp_decode_block(model.0, self.0, n_tokens, result.0) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(())
    }
}

impl Drop for KvState {
    fn drop(&mut self) {
        let mut handle = self.0;
        unsafe {
            let _ = raw::hyp_kvstate_free(&mut handle);
        }
        self.0 = handle;
    }
}

/// Owned step-result handle (caller-owned, reused across decode steps at 2.x).
pub struct StepResult(raw::HypStepResult);

impl StepResult {
    pub fn create() -> Result<Self, Error> {
        let mut handle: raw::HypStepResult = std::ptr::null_mut();
        // SAFETY: `handle` is a writable out-pointer.
        let status = unsafe { raw::hyp_step_result_create(&mut handle) };
        if status != raw::HYP_STATUS_OK {
            return Err(native_error(status));
        }
        Ok(Self(handle))
    }
}

impl Drop for StepResult {
    fn drop(&mut self) {
        let mut handle = self.0;
        unsafe {
            let _ = raw::hyp_step_result_free(&mut handle);
        }
        self.0 = handle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layout_matches_header() {
        assert_eq!(std::mem::size_of::<raw::HypCanaryInfo>(), 224);
        assert_eq!(
            std::mem::offset_of!(raw::HypCanaryInfo, recommended_working_set_bytes),
            32
        );
        assert_eq!(
            std::mem::offset_of!(raw::HypCanaryInfo, mlx_runtime_version),
            64
        );
        assert_eq!(std::mem::offset_of!(raw::HypCanaryInfo, gpu_name), 96);
        assert_eq!(std::mem::size_of::<raw::RusageInfoV4>(), 296);
        assert_eq!(std::mem::offset_of!(raw::RusageInfoV4, wired_size), 56);
        assert_eq!(std::mem::offset_of!(raw::RusageInfoV4, resident_size), 64);
        assert_eq!(std::mem::offset_of!(raw::RusageInfoV4, phys_footprint), 72);
        assert_eq!(
            std::mem::offset_of!(raw::RusageInfoV4, lifetime_max_phys_footprint),
            240
        );
    }

    #[test]
    fn m2_abi_layout_matches_header() {
        // The repr(C) mirrors must match the C header exactly (drift guard).
        assert_eq!(std::mem::size_of::<raw::HypRopeSpec>(), 24);
        assert_eq!(std::mem::offset_of!(raw::HypRopeSpec, theta), 0);
        assert_eq!(
            std::mem::offset_of!(raw::HypRopeSpec, partial_rotary_factor),
            12
        );
        assert_eq!(std::mem::size_of::<raw::HypMoeConfig>(), 12);
        assert_eq!(std::mem::size_of::<raw::HypTokenStream>(), 16);
        assert_eq!(std::mem::offset_of!(raw::HypTokenStream, tokens), 0);
        assert_eq!(std::mem::offset_of!(raw::HypTokenStream, count), 8);
        assert_eq!(std::mem::size_of::<raw::HypStepResultFields>(), 64);
        assert_eq!(
            std::mem::offset_of!(raw::HypStepResultFields, peak_mlx_bytes),
            8
        );
        assert_eq!(
            std::mem::offset_of!(raw::HypStepResultFields, governor_state),
            56
        );
        // HypGeometryParams: the load-bearing pointer field + rope/moe placement.
        assert_eq!(
            std::mem::offset_of!(raw::HypGeometryParams, layer_types),
            16
        );
        assert_eq!(std::mem::offset_of!(raw::HypGeometryParams, rope_local), 56);
        assert_eq!(std::mem::offset_of!(raw::HypGeometryParams, moe), 144);
        assert_eq!(std::mem::size_of::<raw::HypGeometryParams>(), 160);
    }

    #[test]
    fn samples_current_process_memory() {
        let sample = sample_process_memory(std::process::id())
            .expect("proc_pid_rusage should sample the current process");
        assert!(sample.resident_size_bytes > 0);
        assert!(sample.phys_footprint_bytes > 0);
    }

    #[test]
    fn null_canary_output_is_a_typed_error() {
        // SAFETY: null is deliberately passed to exercise the ABI's argument guard.
        let status = unsafe { raw::hyp_runtime_canary(std::ptr::null_mut()) };
        let error = native_error(status);
        assert_eq!(error.status, Status::InvalidArgument);
        assert!(error.message.contains("non-null output pointer"));
    }

    #[test]
    fn model_handle_create_drop_round_trip() {
        let model = Model::create().expect("model create should succeed");
        drop(model); // must not panic / leak.
    }

    #[test]
    fn kvstate_and_step_result_create_drop() {
        let model = Model::create().expect("model create");
        let kv = KvState::create(&model).expect("kvstate create");
        let result = StepResult::create().expect("step result create");
        drop(kv);
        drop(result);
        drop(model);
    }

    #[test]
    fn double_free_is_a_safe_noop() {
        // SAFETY: the handle is owned and freed once here, then freed again to
        // exercise the idempotent-null path (no use-after-free, no UB).
        let mut handle: raw::HypModel = std::ptr::null_mut();
        unsafe {
            assert_eq!(raw::hyp_model_create(&mut handle), raw::HYP_STATUS_OK);
            assert!(!handle.is_null());
            assert_eq!(raw::hyp_model_free(&mut handle), raw::HYP_STATUS_OK);
            assert!(handle.is_null());
            // Second free: handle is null -> idempotent no-op (OK), no UB.
            assert_eq!(raw::hyp_model_free(&mut handle), raw::HYP_STATUS_OK);
        }
    }

    #[test]
    fn null_handle_arguments_are_typed_errors() {
        // SAFETY: null out-pointers exercise the argument guards.
        unsafe {
            let create_status = raw::hyp_model_create(std::ptr::null_mut());
            assert_eq!(native_error(create_status).status, Status::InvalidArgument);

            let free_status = raw::hyp_model_free(std::ptr::null_mut());
            assert_eq!(
                native_error(free_status).status,
                Status::InvalidArgument,
                "free(null* ) must reject, not UB"
            );
        }
    }

    #[test]
    fn wrong_type_handle_is_rejected() {
        // SAFETY: allocate a step-result handle, then pass it where a model is
        // expected — the magic mismatch must produce InvalidArgument, not UB.
        let mut step: raw::HypStepResult = std::ptr::null_mut();
        unsafe {
            assert_eq!(raw::hyp_step_result_create(&mut step), raw::HYP_STATUS_OK);
            let status =
                raw::hyp_model_load(step as raw::HypModel, std::ptr::null(), std::ptr::null());
            let error = native_error(status);
            assert_eq!(error.status, Status::InvalidArgument);
            assert!(error.message.contains("magic mismatch"));
            assert_eq!(raw::hyp_step_result_free(&mut step), raw::HYP_STATUS_OK);
        }
    }

    #[test]
    fn load_prefill_decode_are_unsupported_until_2x() {
        let model = Model::create().expect("model create");
        let kv = KvState::create(&model).expect("kvstate create");
        let result = StepResult::create().expect("step result create");

        let geometry = raw::HypGeometryParams {
            model_type: raw::HYP_GEMMA4_UNIFIED_TEXT,
            hidden_size: 3840,
            intermediate_size: 15360,
            num_hidden_layers: 48,
            layer_types: std::ptr::null(),
            num_attention_heads: 16,
            head_dim_local: 256,
            head_dim_global: 512,
            num_kv_heads_local: 8,
            num_kv_heads_global: 1,
            attention_k_eq_v_global: 1,
            num_kv_shared_layers: 0,
            sliding_window: 1024,
            rope_local: raw::HypRopeSpec {
                theta: 10000.0,
                has_partial_rotary_factor: 0,
                partial_rotary_factor: 0.0,
                proportional: 0,
            },
            rope_global: raw::HypRopeSpec {
                theta: 1000000.0,
                has_partial_rotary_factor: 1,
                partial_rotary_factor: 0.25,
                proportional: 1,
            },
            final_logit_softcapping: 30.0,
            rms_norm_eps: 1e-6,
            attention_bias: 0,
            vocab_size: 262144,
            max_position_embeddings: 262144,
            tie_word_embeddings: 1,
            ple_hidden_per_layer_input: 0,
            ple_vocab_per_layer_input: 0,
            use_double_wide_mlp: 0,
            has_moe: 0,
            moe: raw::HypMoeConfig {
                num_experts: 0,
                top_k: 0,
                moe_intermediate_size: 0,
            },
        };

        let load_error = model.load(&geometry, "/nonexistent/weights").unwrap_err();
        assert_eq!(load_error.status, Status::Unsupported);

        // SAFETY: valid handles; the stub validates then returns Unsupported.
        let tokens = raw::HypTokenStream {
            tokens: std::ptr::null(),
            count: 0,
            is_prompt: 1,
        };
        unsafe {
            let prefill_status = raw::hyp_prefill_chunk(model.0, kv.0, &tokens, result.0);
            assert_eq!(native_error(prefill_status).status, Status::Unsupported);

            let decode_status = raw::hyp_decode_block(model.0, kv.0, 1, result.0);
            assert_eq!(native_error(decode_status).status, Status::Unsupported);
        }
    }

    #[test]
    #[ignore = "requires an unsandboxed M5 with macOS 26.2+ and MLX 0.32.0"]
    fn real_m5_canary() {
        let info = runtime_canary().expect("M5 canary should pass");
        assert!(info.gpu_family >= 1010);
        assert_eq!(info.mlx_compile_version, (0, 32, 0));
        assert_eq!(info.mlx_runtime_version, "0.32.0");
        assert_eq!(info.mlx_probe_value, 4.0);
        assert_eq!(info.metallib_probe_value, 42.0);
    }
}
