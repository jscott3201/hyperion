//! Safe Rust wrapper over Hyperion's narrow native C ABI.
//!
//! This is the only workspace crate permitted to contain `unsafe` code.

use std::{
    error::Error as StdError,
    ffi::CStr,
    fmt,
    os::raw::{c_char, c_int},
};

mod raw {
    use super::{c_char, c_int};

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
}

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
