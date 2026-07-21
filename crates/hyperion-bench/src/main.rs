use std::process::ExitCode;

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("canary") => match hyperion_core::startup_canary() {
            Ok(info) => {
                println!("{}", hyperion_bench::measured_canary_line(&info));
                ExitCode::SUCCESS
            }
            Err(error) => {
                let reason = error.to_string().replace(['\r', '\n'], " ");
                eprintln!("CANARY_FAILED {reason}");
                ExitCode::FAILURE
            }
        },
        Some("--help" | "-h") => {
            println!("usage: hyperion-bench canary");
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: hyperion-bench canary");
            ExitCode::from(64)
        }
    }
}
