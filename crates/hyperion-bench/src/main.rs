use std::process::ExitCode;

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.first().map(String::as_str) {
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
        Some("m1") => match hyperion_bench::m1::run_cli(&arguments[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("M1_FAILED {error}");
                ExitCode::FAILURE
            }
        },
        Some("--help" | "-h") => {
            println!("usage: hyperion-bench canary");
            println!("{}", hyperion_bench::m1::m1_usage());
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("usage: hyperion-bench canary");
            eprintln!("{}", hyperion_bench::m1::m1_usage());
            ExitCode::from(64)
        }
    }
}
