//! Evidence formatting shared by the M0 canary CLI and tests.

use hyperion_core::CanaryInfo;

/// Format one append-ready M0 canary evidence line.
#[must_use]
pub fn measured_canary_line(info: &CanaryInfo) -> String {
    format!(
        concat!(
            "MEASURED {{\"schema\":\"hyperion.canary.v1\",",
            "\"model_scope\":\"{}\",\"abi_version\":{},",
            "\"macos\":\"{}.{}.{}\",\"gpu_family\":{},\"gpu_name\":\"{}\",",
            "\"mlx_compile\":\"{}.{}.{}\",\"mlx_runtime\":\"{}\",",
            "\"recommended_working_set_bytes\":{},\"budget_formula\":",
            "\"min(12GiB,floor(recommended*0.949))\",",
            "\"effective_budget_bytes\":{},\"soft_watermark_bytes\":{},",
            "\"mlx_probe_value\":{:.1},\"metallib_probe_value\":{:.1}}}"
        ),
        hyperion_model::ModelFamily::Gemma4.as_str(),
        info.abi_version,
        info.macos_version.0,
        info.macos_version.1,
        info.macos_version.2,
        info.gpu_family,
        json_escape(&info.gpu_name),
        info.mlx_compile_version.0,
        info.mlx_compile_version.1,
        info.mlx_compile_version.2,
        json_escape(&info.mlx_runtime_version),
        info.recommended_working_set_bytes,
        info.effective_budget_bytes,
        info.soft_watermark_bytes,
        info.mlx_probe_value,
        info.metallib_probe_value,
    )
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            control if control.is_control() => {
                use std::fmt::Write as _;
                write!(escaped, "\\u{:04x}", u32::from(control))
                    .expect("writing to String cannot fail");
            }
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_line_is_single_line_and_machine_readable() {
        let line = measured_canary_line(&CanaryInfo {
            abi_version: 1,
            mlx_compile_version: (0, 32, 0),
            mlx_runtime_version: "0.32.0".to_owned(),
            macos_version: (26, 6, 0),
            gpu_family: 1010,
            recommended_working_set_bytes: 12_713_115_648,
            effective_budget_bytes: 12_064_746_749,
            soft_watermark_bytes: 10_858_272_074,
            mlx_probe_value: 4.0,
            metallib_probe_value: 42.0,
            gpu_name: "Apple M5\nspoof".to_owned(),
        });
        assert!(line.starts_with("MEASURED {"));
        assert!(line.contains("Apple M5\\nspoof"));
        assert_eq!(line.lines().count(), 1);
        assert!(line.ends_with('}'));
    }
}
