use std::process::ExitCode;

fn main() -> ExitCode {
    match std::env::args().nth(1).as_deref() {
        Some("--build-info") => {
            let info = hyperion_server::build_info();
            println!(
                "hyperion family={} native_backends={} serving_available={}",
                info.engine.model_family.as_str(),
                info.engine.native_backend_count,
                info.serving_available
            );
            ExitCode::SUCCESS
        }
        Some("--help" | "-h") => {
            println!("usage: hyperion-server --build-info");
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("UNSUPPORTED: HTTP serving is not exposed before milestone M3");
            ExitCode::from(64)
        }
    }
}
