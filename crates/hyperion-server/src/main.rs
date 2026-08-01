//! hyperion-server: the M3 axum dual-dialect + real SSE server. Loads the
//! model at startup, spawns the dedicated engine thread, and serves the
//! `/v1/*` + `/control/*` routes (06 §Surfaces). localhost-first; a non-loopback
//! bind requires `--token` (fail-closed, B3).
//!
//! Usage:
//!   hyperion-server --model <artifact-dir> [--addr 127.0.0.1:8080]
//!                   [--token <bearer>] [--max-body-mib 32] [--build-info]

use std::process::ExitCode;
use std::sync::Arc;

use hyperion_server::auth::{bind_gate, parse_bind_addr};
use hyperion_server::control::ControlState;
use hyperion_server::engine::{MailboxEngine, engine_thread_loop};
use hyperion_server::prepare::ContextWindow;
use hyperion_server::server::{Server, ServerConfig};
use hyperion_tokenizer::TokenizerHandle;
use hyperion_tokenizer::renderer::ChatTemplate;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(reason) => {
            eprintln!("hyperion-server: {reason}");
            ExitCode::FAILURE
        }
    }
}

/// The default max-tokens when a dialect body omits it (a conservative server
/// default; the model card allows more, but a server shouldn't default to the
/// full window — keep generations bounded).
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Run the server. Parses args, applies the bind-gate, loads the model, spawns
/// the engine thread, and serves until SIGTERM/`/control/shutdown`.
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--build-info") {
        let info = hyperion_server::build_info();
        println!(
            "hyperion family={} native_backends={} serving_available={}",
            info.engine.model_family.as_str(),
            info.engine.native_backend_count,
            info.serving_available,
        );
        return Ok(());
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "usage: hyperion-server --model <artifact-dir> [--addr 127.0.0.1:8080] [--token <bearer>] [--max-body-mib 32]"
        );
        return Ok(());
    }

    let model_dir = arg_value(&args, "--model")
        .ok_or_else(|| "--model <artifact-dir> is required".to_string())?;
    let addr_str = arg_value(&args, "--addr").unwrap_or_else(|| "127.0.0.1:8080".to_string());
    let token = arg_value(&args, "--token");
    let (addr, port) = parse_bind_addr(&addr_str)?;

    // B3: the bind-gate. Non-loopback without a token → refuse to start.
    let _gate = bind_gate(addr, token.as_deref())?;
    // (BindGate::Loopback ⇒ no auth; Authenticated ⇒ bearer enforced. Both OK
    // to proceed; bind_gate already refused the fail-closed case.)

    // Load the geometry from the artifact's config.json.
    let config_path = std::path::Path::new(&model_dir).join("config.json");
    let config_str =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read {config_path:?}: {e}"))?;
    let geometry = hyperion_model::geometry::Geometry::from_config_str(&config_str)
        .map_err(|e| format!("parse geometry: {e}"))?;
    let context: ContextWindow = geometry.max_position_embeddings;

    // The model id (echoed in /v1/models + SSE frames) — the artifact dir name.
    let model_id = std::path::Path::new(&model_dir)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("hyperion")
        .to_string();

    // Load the tokenizer + chat template from the same artifact dir.
    let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer =
        TokenizerHandle::from_file(&tokenizer_path).map_err(|e| format!("load tokenizer: {e}"))?;
    let template = ChatTemplate::from_artifact(std::path::Path::new(&model_dir), Some(&tokenizer))
        .map_err(|e| format!("load chat template: {e}"))?;

    // Spawn the dedicated engine thread. The `!Send` `Engine` is loaded **on**
    // the engine thread (it can't be moved across threads); `engine_thread_loop`
    // loads it from `geometry` + `model_dir` and reports success via `loaded`.
    let (mailbox, rx) = MailboxEngine::channel();
    let (loaded_tx, loaded_rx) = std::sync::mpsc::channel();
    let engine_geometry = geometry.clone();
    let engine_model_dir = model_dir.clone();
    let engine_handle = std::thread::Builder::new()
        .name("hyperion-engine".into())
        .spawn(move || engine_thread_loop(engine_geometry, engine_model_dir, rx, loaded_tx))
        .map_err(|e| format!("spawn engine thread: {e}"))?;
    // Wait for the engine thread to finish loading before serving (a 503
    // not-ready gate would be the async alternative; PR B loads-at-startup).
    loaded_rx
        .recv()
        .map_err(|e| format!("engine thread exited before load: {e}"))?
        .map_err(|e| format!("load model: {e}"))?;

    // The shared ops state. The model is loaded → ready.
    let control = ControlState::new(&model_id);
    control.set_ready(true);

    let server = Server::new(ServerConfig {
        engine: Arc::new(mailbox),
        template,
        tokenizer,
        control: control.clone(),
        bearer: token,
        context,
        model_id: model_id.clone(),
        default_max_tokens: DEFAULT_MAX_TOKENS,
    });
    let router = server.router();
    let addr_bind = std::net::SocketAddr::new(addr, port);

    // Serve on a multi-thread runtime (the blocking engine tasks + SSE drains
    // don't starve each other). Ends on `/control/shutdown` (the future
    // resolves via the shutdown poll).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("build tokio runtime: {e}"))?;
    let serve_result: Result<(), String> = rt.block_on(async move {
        let listener = match tokio::net::TcpListener::bind(addr_bind).await {
            Ok(l) => l,
            Err(e) => return Err(format!("bind {addr_bind}: {e}")),
        };
        eprintln!("hyperion-server: serving {addr_bind} (model={model_id})");
        let serve = axum::serve(listener, router);
        // Poll the shutdown flag; a full graceful drain is a later refinement
        // (PR B stops accepting + lets the in-flight request finish).
        let shutdown = async {
            loop {
                if control.is_shutdown() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            }
        };
        tokio::select! {
            res = serve => {
                if let Err(e) = res {
                    eprintln!("hyperion-server: serve error: {e}");
                }
            }
            _ = shutdown => {
                eprintln!("hyperion-server: shutdown requested; draining");
            }
        }
        // The engine thread ends when the mailbox sender drops (it doesn't in
        // PR B — the Arc is alive until process exit). The native handles free
        // at process exit; PR B leaks the join on shutdown (the process is
        // ending). Drop the handle to signal intent.
        drop(engine_handle);
        Ok::<(), String>(())
    });
    serve_result.map_err(|e| format!("serve: {e}"))
}

/// Extract the value for `--flag` from the args (the next token). Returns
/// `None` if the flag isn't present.
fn arg_value(args: &[String], flag: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(a) = iter.next() {
        if a == flag {
            return iter.next().cloned();
        }
    }
    None
}
