//! Reproducible controller and validators for the stock-mlx-lm M1 baseline.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error as StdError,
    ffi::OsStr,
    fmt,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Component, Path, PathBuf},
    process::{ChildStdin, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use hyperion_ffi::ProcessMemorySample;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const CONTROLLER_SCHEMA: &str = "hyperion.m1-controller-event.v1";
const WORKER_SCHEMA: &str = "hyperion.m1-worker-event.v1";
const TRIAL_SCHEMA: &str = "hyperion.m1-trial.v1";
const OS_SAMPLE_SCHEMA: &str = "hyperion.m1-os-sample.v1";
const CORPUS_SCHEMA: &str = "hyperion.m1-corpus.v1";
const ORACLE_LOCK_SHA256: &str = "b3603b4ebbc7f5883afe3d8cc10fc1767239837f5256985bd591b777c993dbaf";
const ORACLE_PYTHON: &str = "3.12.13";
const MLX_VERSION: &str = "0.32.0";
const MLX_METAL_VERSION: &str = "0.32.0";
const MLX_LM_VERSION: &str = "0.31.3";
const MLX_LM_COMMIT: &str = "8239c72de5a0e42c539e30489021db73c7fe258c";
const MLX_TREE_SHA256: &str = "bacebd4f46680155a129301ffefc516402142183584f2b47673bc91b561f0cd9";
const MLX_METAL_TREE_SHA256: &str =
    "628a99548b65855148fb03f71cac83ce46eae42140f119fa8d1b51285c2abefd";
const MLX_LM_TREE_SHA256: &str = "40dc49399a07cdf22e3516070cfe222e89ec2f0ff29cd6e257e1b069edc3472f";
const MLX_LM_SERVER_SHA256: &str =
    "cdfcb4ac848636f9927851a0ec7a951584526530cb7832ba58049e4a9144db8b";
const MLX_LM_PACKAGE_SHA256: &str =
    "f9ffa88772d26e537a98aa39ab16488a7a0d13cc1fac5d665376132c94b49608";
const MLX_LM_GENERATE_SHA256: &str =
    "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f";
const PYTHON_EXECUTABLE_SHA256: &str =
    "01564940172b2811e1f39a4dc90e84c7a26a19cf071bbc5de67e456d82627bec";
const PYTHON_RUNTIME_TREE_SHA256: &str =
    "01a580d385a91f4b8bc195c8b2f56c4c2d156f6c1e1ad8768fc4501987c4e12f";
const PYTHON_RUNTIME_FILE_COUNT: u64 = 1_897;
const SITE_PACKAGES_TREE_SHA256: &str =
    "db258e22404a3937d46d72ff44083400aafcf34636b8444a91a29c858b297006";
const SITE_PACKAGES_FILE_COUNT: u64 = 5_470;
const RUN_MANIFEST_SCHEMA: &str = "hyperion.m1-run-manifest.v1";
const SCHEDULE_SCHEMA: &str = "hyperion.m1-schedule.v1";
const COMMAND_SCHEMA: &str = "hyperion.m1-controller-command.v1";
const GENERATED_TOKENS: u32 = 1_025;
const NIGHTLY_GENERATED_TOKENS: u32 = 129;
const WARMUPS: u32 = 1;
const TRIALS: u32 = 5;
const SAMPLE_PERIOD: Duration = Duration::from_millis(25);

/// M1 controller failure.
#[derive(Debug)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl StdError for Error {}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug)]
struct ModelSpec {
    key: &'static str,
    label: &'static str,
    manifest_sha256: &'static str,
    payload_tree_sha256: &'static str,
    payload_file_count: u64,
    default_relative_path: &'static str,
    primary_env: &'static str,
    fallback_env: Option<&'static str>,
}

const MODEL_12B: ModelSpec = ModelSpec {
    key: "12b",
    label: "gemma-4-12B-QAT-Q4-g64-affine",
    manifest_sha256: "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144",
    payload_tree_sha256: "60386542c026e72aa7b8b4a3ffb3e2356fd3e80c3d54939ad75d932d59bff2d7",
    payload_file_count: 9,
    default_relative_path: "artifacts/models/gemma4-12b-qat-mlx-g64-b4",
    primary_env: "HYPERION_M1_12B_ORACLE_MODEL",
    fallback_env: Some("HYPERION_M0_ORACLE_MODEL"),
};

const MODEL_E4B: ModelSpec = ModelSpec {
    key: "e4b",
    label: "gemma-4-E4B-QAT-Q4-g64-affine",
    manifest_sha256: "9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0",
    payload_tree_sha256: "99e6875ffb1bf4eae37242c7c0736a43b45b38dd0088ed671551283c0e268f16",
    payload_file_count: 8,
    default_relative_path: "artifacts/models/gemma4-e4b-qat-mlx-g64-b4",
    primary_env: "HYPERION_M1_E4B_ORACLE_MODEL",
    fallback_env: None,
};

#[derive(Debug)]
struct RunCellArgs {
    model: ModelSpec,
    context: u32,
    output: PathBuf,
    arm: String,
    wired_limit: String,
    generated_tokens: u32,
    warmups: u32,
    trials: u32,
    run_manifest: PathBuf,
}

#[derive(Debug)]
struct BeginRunArgs {
    output_dir: PathBuf,
    run_id: String,
}

#[derive(Clone, Debug, Deserialize)]
struct RunManifest {
    schema: String,
    run_id: String,
    session_id: String,
    created_unix_ns: u64,
    source_commit: String,
    executable_sha256: String,
    native_mlx_dylib_sha256: String,
    canary_metallib_sha256: String,
    cargo_lock_sha256: String,
    oracle_lock_sha256: String,
    corpus_manifest_sha256: String,
    schedule_sha256: String,
    worker_sha256: String,
    server_worker_sha256: String,
    oracle_identity_source_sha256: String,
    oracle_launcher_sha256: String,
    model_identity_source_sha256: String,
    preflight_log_sha256: String,
    oracle_verification_pre_sha256: String,
    model_verification_pre_sha256: String,
    machine_profile_sha256: String,
    native_canary: RunCanary,
}

#[derive(Clone, Debug, Deserialize)]
struct RunCanary {
    macos: String,
    gpu_name: String,
    mlx_runtime: String,
    recommended_working_set_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct CorpusManifest {
    schema: String,
    fixtures: Vec<CorpusFixture>,
}

#[derive(Debug, Deserialize)]
struct CorpusFixture {
    model_key: String,
    model_label: String,
    model_manifest_sha256: String,
    tokenizer_sha256: String,
    template_sha256: String,
    target_tokens: u32,
    actual_tokens: u32,
    rendered_file: String,
    rendered_sha256: String,
    token_file: String,
    token_sha256: String,
    token_encoding: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
struct TrialKey {
    phase: String,
    trial_index: u32,
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    elapsed_ns: u64,
    trial: Option<usize>,
    boundary: Option<SampleBoundary>,
    memory: ProcessMemorySample,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrialStage {
    Prepared,
    Running,
    Result,
    PostCleanup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TraceWorkerStage {
    AwaitingStart,
    AwaitingModel,
    Trials,
    Terminated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OpenTrial {
    index: usize,
    stage: TrialStage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SampleBoundary {
    PreTrial,
    PostTrial,
}

impl SampleBoundary {
    const fn as_str(self) -> &'static str {
        match self {
            Self::PreTrial => "pre_trial",
            Self::PostTrial => "post_trial",
        }
    }
}

#[derive(Debug, Default)]
struct TraceCounts {
    worker_start: u32,
    worker_end: u32,
    warmups: u32,
    measured: u32,
    failure: u32,
    measured_hashes: BTreeSet<String>,
    trial_start_pending: u32,
    trial_start: u32,
    trial_post_cleanup_pending: u32,
    trial_end: u32,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct TrialMetrics {
    trial_index: u32,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    ttft_ns: u64,
    itl_p50_ns: u64,
    itl_p95_ns: u64,
    itl_p99_ns: u64,
    mlx_active_end_bytes: u64,
    mlx_cache_end_bytes: u64,
    mlx_peak_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct OsTrialMetrics {
    trial_index: u32,
    sample_count: u64,
    pre_trial: ProcessMemorySample,
    post_trial: ProcessMemorySample,
    first: ProcessMemorySample,
    last: ProcessMemorySample,
    max_wired_size_bytes: u64,
    max_resident_size_bytes: u64,
    max_phys_footprint_bytes: u64,
    max_interval_phys_footprint_bytes: u64,
    max_lifetime_phys_footprint_bytes: u64,
    pageins_first: u64,
    pageins_last: u64,
}

#[derive(Clone, Debug)]
struct OsSampleEvidence {
    elapsed_ns: u64,
    phase: Option<String>,
    trial_index: Option<u32>,
    boundary: Option<String>,
    memory: ProcessMemorySample,
}

#[derive(Clone, Debug)]
struct Cell {
    path: PathBuf,
    sha256: String,
    run_id: String,
    session_id: String,
    run_manifest_sha256: String,
    schedule_sha256: String,
    corpus_manifest_sha256: String,
    model_verification_sha256: String,
    worker_sha256: String,
    executable_sha256: String,
    source_commit: String,
    model_key: String,
    context_tokens: u32,
    generated_tokens: u32,
    arm: String,
    requested_wired_limit_bytes: u64,
    recommended_working_set_bytes: u64,
    started_unix_ns: u64,
    finished_unix_ns: u64,
    output_token_sha256: Option<String>,
    trials: Vec<TrialMetrics>,
    os_trials: Vec<OsTrialMetrics>,
    validation_errors: Vec<String>,
    failure_recorded: bool,
    uncontrolled_oom: bool,
    trace_complete: bool,
    valid: bool,
}

#[derive(Debug, Serialize)]
struct Distribution {
    min: f64,
    median: f64,
    max: f64,
}

/// Dispatch an `m1` CLI invocation after the top-level `m1` token.
pub fn run_cli(arguments: &[String]) -> Result<(), Error> {
    match arguments.first().map(String::as_str) {
        Some("begin-run") => begin_run(parse_begin_run(&arguments[1..])?),
        Some("run-cell") => run_cell(parse_run_cell(&arguments[1..])?),
        Some("verify-trace") => {
            let path = arguments
                .get(1)
                .ok_or_else(|| Error::new("usage: hyperion-bench m1 verify-trace PATH"))?;
            if arguments.len() != 2 {
                return Err(Error::new("usage: hyperion-bench m1 verify-trace PATH"));
            }
            verify_trace(Path::new(path))
        }
        Some("summarize") => {
            let directory = parse_directory_argument(&arguments[1..], "summarize")?;
            summarize(&directory)
        }
        Some("select-budget") => {
            let directory = parse_directory_argument(&arguments[1..], "select-budget")?;
            select_budget(&directory)
        }
        Some("check-acca") => {
            let directory = parse_directory_argument(&arguments[1..], "check-acca")?;
            check_acca(&directory)
        }
        Some("verify-run") => {
            let directory = parse_directory_argument(&arguments[1..], "verify-run")?;
            verify_run(&directory)
        }
        _ => Err(Error::new(m1_usage())),
    }
}

/// M1 CLI usage.
#[must_use]
pub fn m1_usage() -> &'static str {
    concat!(
        "usage:\n",
        "  hyperion-bench m1 begin-run --output-dir benchmarks/raw/m1/RUN --run-id RUN\n",
        "  hyperion-bench m1 run-cell --model 12b|e4b --context TOKENS --arm NAME --wired-limit default|BYTES --run-manifest benchmarks/raw/m1/RUN/run-manifest.json --output benchmarks/raw/m1/RUN/FILE.jsonl\n",
        "  hyperion-bench m1 verify-trace PATH\n",
        "  hyperion-bench m1 summarize --input-dir DIR\n",
        "  hyperion-bench m1 select-budget --input-dir DIR\n",
        "  hyperion-bench m1 check-acca --input-dir DIR\n",
        "  hyperion-bench m1 verify-run --input-dir benchmarks/raw/m1/RUN"
    )
}

fn parse_begin_run(arguments: &[String]) -> Result<BeginRunArgs, Error> {
    if arguments.len() != 4 {
        return Err(Error::new(
            "usage: hyperion-bench m1 begin-run --output-dir benchmarks/raw/m1/RUN --run-id RUN",
        ));
    }
    let mut output_dir = None;
    let mut run_id = None;
    for pair in arguments.chunks_exact(2) {
        match pair[0].as_str() {
            "--output-dir" if output_dir.is_none() => output_dir = Some(PathBuf::from(&pair[1])),
            "--run-id" if run_id.is_none() => run_id = Some(pair[1].clone()),
            flag => {
                return Err(Error::new(format!(
                    "unknown or duplicate begin-run flag: {flag}"
                )));
            }
        }
    }
    let output_dir = output_dir.ok_or_else(|| Error::new("begin-run lacks --output-dir"))?;
    let run_id = run_id.ok_or_else(|| Error::new("begin-run lacks --run-id"))?;
    if run_id.is_empty()
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || output_dir.is_absolute()
        || output_dir.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || output_dir.parent() != Some(Path::new("benchmarks/raw/m1"))
        || output_dir.file_name() != Some(OsStr::new(&run_id))
    {
        return Err(Error::new(
            "begin-run output must be exactly benchmarks/raw/m1/RUN and match a safe run ID",
        ));
    }
    Ok(BeginRunArgs { output_dir, run_id })
}

fn parse_directory_argument(arguments: &[String], command: &str) -> Result<PathBuf, Error> {
    if arguments.len() != 2 || arguments[0] != "--input-dir" {
        return Err(Error::new(format!(
            "usage: hyperion-bench m1 {command} --input-dir DIR"
        )));
    }
    Ok(PathBuf::from(&arguments[1]))
}

fn parse_run_cell(arguments: &[String]) -> Result<RunCellArgs, Error> {
    let mut flags = BTreeMap::<String, String>::new();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        if !flag.starts_with("--") {
            return Err(Error::new(format!(
                "unexpected positional argument: {flag}"
            )));
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| Error::new(format!("missing value for {flag}")))?;
        if flags.insert(flag.clone(), value.clone()).is_some() {
            return Err(Error::new(format!("duplicate argument: {flag}")));
        }
        index += 2;
    }
    let allowed = BTreeSet::from([
        "--model",
        "--context",
        "--output",
        "--arm",
        "--wired-limit",
        "--generated-tokens",
        "--warmups",
        "--trials",
        "--run-manifest",
    ]);
    for flag in flags.keys() {
        if !allowed.contains(flag.as_str()) {
            return Err(Error::new(format!("unknown run-cell argument: {flag}")));
        }
    }
    let required = |name: &str| {
        flags
            .get(name)
            .cloned()
            .ok_or_else(|| Error::new(format!("missing required argument: {name}")))
    };
    let model = match required("--model")?.as_str() {
        "12b" => MODEL_12B,
        "e4b" => MODEL_E4B,
        other => return Err(Error::new(format!("unsupported M1 model: {other}"))),
    };
    let context = parse_positive_u32(&required("--context")?, "context")?;
    if ![512, 1_024, 4_096, 8_192, 16_384, 32_768, 131_072].contains(&context) {
        return Err(Error::new(format!(
            "unsupported M1 context {context}; use 512/1024/4096/8192/16384/32768/131072"
        )));
    }
    let arm = required("--arm")?;
    if arm.is_empty()
        || !arm
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(Error::new(
            "arm must contain only ASCII letters, digits, '-' or '_'",
        ));
    }
    let wired_limit = required("--wired-limit")?;
    if wired_limit != "default" {
        let limit = wired_limit
            .parse::<u64>()
            .map_err(|_| Error::new("wired-limit must be 'default' or positive bytes"))?;
        if limit == 0 {
            return Err(Error::new("wired-limit bytes must be positive"));
        }
    }
    let generated_tokens = flags
        .get("--generated-tokens")
        .map_or(Ok(GENERATED_TOKENS), |value| {
            parse_positive_u32(value, "generated-tokens")
        })?;
    let warmups = flags
        .get("--warmups")
        .map_or(Ok(WARMUPS), |value| parse_positive_u32(value, "warmups"))?;
    let trials = flags
        .get("--trials")
        .map_or(Ok(TRIALS), |value| parse_positive_u32(value, "trials"))?;
    let gating_cardinality = generated_tokens == GENERATED_TOKENS && context != 131_072;
    let nightly_cardinality =
        generated_tokens == NIGHTLY_GENERATED_TOKENS && context == 512 && arm == "nightly-512x128";
    let stretch_cardinality = generated_tokens == NIGHTLY_GENERATED_TOKENS
        && context == 131_072
        && arm == "stretch-128k"
        && (1..TRIALS).contains(&trials);
    let standard_cardinality = (gating_cardinality || nightly_cardinality) && trials == TRIALS;
    if (!standard_cardinality && !stretch_cardinality) || warmups != WARMUPS {
        return Err(Error::new(
            "run-cell requires one warmup/five trials for 1025-ID or nightly rows, or an explicit 128K 129-ID stretch row with one to four trials",
        ));
    }
    let output = validate_raw_output_path(&required("--output")?)?;
    let run_manifest = validate_run_manifest_path(&required("--run-manifest")?)?;
    Ok(RunCellArgs {
        model,
        context,
        output,
        arm,
        wired_limit,
        generated_tokens,
        warmups,
        trials,
        run_manifest,
    })
}

fn validate_run_manifest_path(value: &str) -> Result<PathBuf, Error> {
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || !path.starts_with("benchmarks/raw/m1")
        || path.file_name() != Some(OsStr::new("run-manifest.json"))
    {
        return Err(Error::new(
            "run manifest must be a relative benchmarks/raw/m1/*/run-manifest.json path without '..'",
        ));
    }
    Ok(path.to_owned())
}

fn parse_positive_u32(value: &str, label: &str) -> Result<u32, Error> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| Error::new(format!("{label} must be a positive integer")))?;
    if parsed == 0 {
        return Err(Error::new(format!("{label} must be positive")));
    }
    Ok(parsed)
}

fn validate_raw_output_path(value: &str) -> Result<PathBuf, Error> {
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
        || !path.starts_with("benchmarks/raw/m1")
        || path.extension() != Some(OsStr::new("jsonl"))
    {
        return Err(Error::new(
            "output must be a relative benchmarks/raw/m1/*.jsonl path without '..'",
        ));
    }
    Ok(path.to_owned())
}

fn begin_run(arguments: BeginRunArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    ensure_clean_worktree(&repo)?;
    let run_root = repo.join(&arguments.output_dir);
    if !run_root.is_dir() {
        return Err(Error::new(format!(
            "begin-run output directory does not exist: {}",
            run_root.display()
        )));
    }
    let manifest_path = run_root.join("run-manifest.json");
    if manifest_path.exists() {
        return Err(Error::new(format!(
            "refusing to overwrite run manifest: {}",
            manifest_path.display()
        )));
    }
    let preflight_log = run_root.join("preflight/preflight.log");
    let oracle_receipt = run_root.join("preflight/oracle-verification.log");
    let model_receipt = run_root.join("preflight/model-verification.log");
    for required in [&preflight_log, &oracle_receipt, &model_receipt] {
        if !required.is_file() {
            return Err(Error::new(format!(
                "begin-run lacks required preflight receipt: {}",
                required.display()
            )));
        }
    }
    verify_preflight_receipts(&repo, &run_root)?;

    let executable = env::current_exe()?;
    let mlx_root =
        PathBuf::from(env::var("MLX_ROOT").unwrap_or_else(|_| "/opt/homebrew/opt/mlx".to_owned()));
    let native_mlx_dylib = mlx_root.join("lib/libmlx.dylib");
    let canary_metallib = executable
        .parent()
        .ok_or_else(|| Error::new("benchmark executable has no parent"))?
        .join("hyperion_canary.metallib");
    let source_commit = command_stdout(&repo, "git", &["rev-parse", "HEAD"])?;
    let architecture = command_stdout(&repo, "uname", &["-m"])?;
    if architecture != "arm64" {
        return Err(Error::new(format!(
            "M1 measurement requires arm64, found {architecture}"
        )));
    }
    let machine_model = command_stdout(&repo, "sysctl", &["-n", "hw.model"])?;
    let physical_memory_bytes = command_stdout(&repo, "sysctl", &["-n", "hw.memsize"])?
        .parse::<u64>()
        .map_err(|_| Error::new("sysctl hw.memsize was not u64"))?;
    let canary = hyperion_core::startup_canary()
        .map_err(|error| Error::new(format!("run-bound native canary failed: {error}")))?;
    if canary.gpu_name != "Apple M5" || canary.mlx_runtime_version != MLX_VERSION {
        return Err(Error::new(format!(
            "M1 requires Apple M5 with MLX {MLX_VERSION}; found {} with {}",
            canary.gpu_name, canary.mlx_runtime_version
        )));
    }
    let macos = format!(
        "{}.{}.{}",
        canary.macos_version.0, canary.macos_version.1, canary.macos_version.2
    );
    let power = power_snapshot(&repo);
    let thermal = thermal_snapshot(&repo);
    let created_unix_ns = unix_time_ns()?;
    let native_canary = json!({
        "macos": macos,
        "gpu_name": canary.gpu_name,
        "gpu_family": canary.gpu_family,
        "mlx_compile": format!("{}.{}.{}", canary.mlx_compile_version.0, canary.mlx_compile_version.1, canary.mlx_compile_version.2),
        "mlx_runtime": canary.mlx_runtime_version,
        "recommended_working_set_bytes": canary.recommended_working_set_bytes,
        "effective_budget_bytes": canary.effective_budget_bytes,
        "soft_watermark_bytes": canary.soft_watermark_bytes,
        "mlx_probe_value": canary.mlx_probe_value,
        "metallib_probe_value": canary.metallib_probe_value,
    });
    let machine_profile = json!({
        "architecture": architecture,
        "machine_model": machine_model,
        "physical_memory_bytes": physical_memory_bytes,
        "native_canary": native_canary,
        "power": power,
        "thermal": thermal,
    });
    let machine_profile_sha256 = sha256_bytes(&serde_json::to_vec(&machine_profile)?);
    let schedule_path = repo.join("benchmarks/m1/schedule.json");
    let schedule: Value = serde_json::from_slice(&fs::read(&schedule_path)?)?;
    if schedule.get("schema").and_then(Value::as_str) != Some(SCHEDULE_SCHEMA) {
        return Err(Error::new("unsupported M1 schedule schema"));
    }
    let session_id = format!("{}-{created_unix_ns}", arguments.run_id);
    let output = json!({
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": arguments.run_id,
        "session_id": session_id,
        "created_unix_ns": created_unix_ns,
        "source_commit": source_commit,
        "worktree_clean": true,
        "executable_sha256": sha256_file(&executable)?,
        "native_mlx_dylib_sha256": sha256_file(&native_mlx_dylib)?,
        "canary_metallib_sha256": sha256_file(&canary_metallib)?,
        "cargo_lock_sha256": sha256_file(&repo.join("Cargo.lock"))?,
        "oracle_lock_sha256": sha256_file(&repo.join("oracle/uv.lock"))?,
        "corpus_manifest_sha256": sha256_file(&repo.join("benchmarks/m1/corpus/manifest.json"))?,
        "schedule_sha256": sha256_file(&schedule_path)?,
        "worker_sha256": sha256_file(&repo.join("oracle/m1_bench_worker.py"))?,
        "server_worker_sha256": sha256_file(&repo.join("oracle/m1_server_smoke.py"))?,
        "oracle_identity_source_sha256": sha256_file(&repo.join("oracle/oracle_identity.py"))?,
        "oracle_launcher_sha256": sha256_file(&repo.join("oracle/isolated_oracle.py"))?,
        "model_identity_source_sha256": sha256_file(&repo.join("oracle/model_identity.py"))?,
        "preflight_log_sha256": sha256_file(&preflight_log)?,
        "oracle_verification_pre_sha256": sha256_file(&oracle_receipt)?,
        "model_verification_pre_sha256": sha256_file(&model_receipt)?,
        "rustc": command_stdout(&repo, "rustc", &["--version"] )?,
        "cargo": command_stdout(&repo, "cargo", &["--version"] )?,
        "build_environment": {
            "CARGO_TARGET_DIR": "target/m1-release",
            "MLX_ROOT": redact_machine_path(path_text(&mlx_root)?, &repo),
            "RUSTFLAGS": null,
            "CARGO_ENCODED_RUSTFLAGS": null,
            "CARGO_PROFILE_OVERRIDES": null,
        },
        "worker_environment": worker_environment(),
        "machine_profile_sha256": machine_profile_sha256,
        "machine_profile": machine_profile,
        "native_canary": native_canary,
    });
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)?;
    serde_json::to_writer_pretty(&mut file, &output)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!(
        "M1_RUN_BEGUN run={} manifest={} sha256={}",
        arguments.run_id,
        arguments.output_dir.join("run-manifest.json").display(),
        sha256_file(&manifest_path)?
    );
    Ok(())
}

fn power_snapshot(repo: &Path) -> Value {
    let output = Command::new("pmset")
        .args(["-g", "batt"])
        .current_dir(repo)
        .output();
    let Ok(output) = output else {
        return json!({"available": false});
    };
    if !output.status.success() {
        return json!({"available": false});
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let source = text
        .lines()
        .next()
        .and_then(|line| line.split('\'').nth(1))
        .filter(|value| matches!(*value, "AC Power" | "Battery Power" | "UPS Power"))
        .unwrap_or("unknown");
    let percent = text.split_whitespace().find_map(|token| {
        token
            .trim_end_matches(['%', ';'])
            .parse::<u8>()
            .ok()
            .filter(|value| *value <= 100)
    });
    let lower = text.to_ascii_lowercase();
    let state = ["discharging", "charging", "charged", "finishing charge"]
        .into_iter()
        .find(|candidate| lower.contains(candidate))
        .unwrap_or("unknown");
    json!({"available": true, "source": source, "percent": percent, "state": state})
}

fn thermal_snapshot(repo: &Path) -> Value {
    let output = Command::new("pmset")
        .args(["-g", "therm"])
        .current_dir(repo)
        .output();
    let Ok(output) = output else {
        return json!({"available": false});
    };
    if !output.status.success() {
        return json!({"available": false});
    }
    json!({
        "available": true,
        "output_sha256": sha256_bytes(&output.stdout),
        "line_count": String::from_utf8_lossy(&output.stdout).lines().count(),
    })
}

fn run_cell(arguments: RunCellArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    ensure_clean_worktree(&repo)?;
    let source_commit = command_stdout(&repo, "git", &["rev-parse", "HEAD"])?;
    let executable = env::current_exe()?;
    let mlx_root =
        PathBuf::from(env::var("MLX_ROOT").unwrap_or_else(|_| "/opt/homebrew/opt/mlx".to_owned()));
    let native_mlx_dylib = mlx_root.join("lib/libmlx.dylib");
    let canary_metallib = executable
        .parent()
        .ok_or_else(|| Error::new("benchmark executable has no parent"))?
        .join("hyperion_canary.metallib");
    let run_manifest_path = repo.join(&arguments.run_manifest);
    let run_manifest_sha256 = sha256_file(&run_manifest_path)?;
    let run_manifest: RunManifest = serde_json::from_slice(&fs::read(&run_manifest_path)?)?;
    let run_root = run_manifest_path
        .parent()
        .ok_or_else(|| Error::new("run manifest has no parent"))?;
    let output_path = repo.join(&arguments.output);
    if run_manifest.schema != RUN_MANIFEST_SCHEMA
        || run_manifest.run_id.is_empty()
        || run_manifest.session_id.is_empty()
        || run_manifest.created_unix_ns == 0
        || run_manifest.machine_profile_sha256.len() != 64
        || run_manifest.source_commit != source_commit
        || run_manifest.executable_sha256 != sha256_file(&executable)?
        || run_manifest.native_mlx_dylib_sha256 != sha256_file(&native_mlx_dylib)?
        || run_manifest.canary_metallib_sha256 != sha256_file(&canary_metallib)?
        || run_manifest.cargo_lock_sha256 != sha256_file(&repo.join("Cargo.lock"))?
        || run_manifest.oracle_lock_sha256 != ORACLE_LOCK_SHA256
        || run_manifest.worker_sha256 != sha256_file(&repo.join("oracle/m1_bench_worker.py"))?
        || run_manifest.server_worker_sha256
            != sha256_file(&repo.join("oracle/m1_server_smoke.py"))?
        || run_manifest.oracle_identity_source_sha256
            != sha256_file(&repo.join("oracle/oracle_identity.py"))?
        || run_manifest.oracle_launcher_sha256
            != sha256_file(&repo.join("oracle/isolated_oracle.py"))?
        || run_manifest.model_identity_source_sha256
            != sha256_file(&repo.join("oracle/model_identity.py"))?
        || run_manifest.schedule_sha256 != sha256_file(&repo.join("benchmarks/m1/schedule.json"))?
        || run_manifest.preflight_log_sha256
            != sha256_file(&run_root.join("preflight/preflight.log"))?
        || run_manifest.oracle_verification_pre_sha256
            != sha256_file(&run_root.join("preflight/oracle-verification.log"))?
        || !output_path.starts_with(run_root)
    {
        return Err(Error::new(
            "run manifest does not bind the current commit, executable, locks, schedule, workers, or output root",
        ));
    }
    let schedule: Value =
        serde_json::from_slice(&fs::read(repo.join("benchmarks/m1/schedule.json"))?)?;
    if schedule.get("schema").and_then(Value::as_str) != Some(SCHEDULE_SCHEMA) {
        return Err(Error::new("unsupported M1 schedule schema"));
    }
    let preflight_model_receipt = run_root.join("preflight/model-verification.log");
    if sha256_file(&preflight_model_receipt)? != run_manifest.model_verification_pre_sha256 {
        return Err(Error::new(
            "preflight full-model verification receipt differs from the run manifest",
        ));
    }

    let corpus_dir = repo.join("benchmarks/m1/corpus");
    let corpus_manifest_path = corpus_dir.join("manifest.json");
    let corpus_manifest_sha256 = sha256_file(&corpus_manifest_path)?;
    let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&corpus_manifest_path)?)?;
    if manifest.schema != CORPUS_SCHEMA {
        return Err(Error::new("unsupported M1 corpus manifest schema"));
    }
    if corpus_manifest_sha256 != run_manifest.corpus_manifest_sha256 {
        return Err(Error::new(
            "M1 corpus manifest differs from the run-bound identity",
        ));
    }
    let fixture = manifest
        .fixtures
        .iter()
        .find(|fixture| {
            fixture.model_key == arguments.model.key && fixture.target_tokens == arguments.context
        })
        .ok_or_else(|| Error::new("M1 corpus manifest does not contain the requested cell"))?;
    validate_fixture(fixture, arguments.model, arguments.context, &corpus_dir)?;

    let model_path = model_path(&repo, arguments.model);
    let actual_model_manifest_sha256 = sha256_file(&model_path.join("SHA256SUMS"))?;
    if actual_model_manifest_sha256 != arguments.model.manifest_sha256 {
        return Err(Error::new(format!(
            "model manifest mismatch: expected {}, got {actual_model_manifest_sha256}",
            arguments.model.manifest_sha256
        )));
    }
    let oracle_lock_path = repo.join("oracle/uv.lock");
    if sha256_file(&oracle_lock_path)? != ORACLE_LOCK_SHA256 {
        return Err(Error::new(
            "oracle/uv.lock differs from accepted M1 protocol",
        ));
    }
    let python = repo.join("oracle/.venv/bin/python");
    let worker = repo.join("oracle/m1_bench_worker.py");
    if !python.is_file() || !worker.is_file() {
        return Err(Error::new(
            "locked oracle or M1 worker is missing; run scripts/setup-oracle.sh",
        ));
    }

    let output_parent = output_path
        .parent()
        .ok_or_else(|| Error::new("raw output has no parent"))?;
    fs::create_dir_all(output_parent)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .map_err(|error| {
            Error::new(format!(
                "refusing to overwrite raw evidence {}: {error}",
                output_path.display()
            ))
        })?;
    let stderr_path = output_path.with_extension("stderr.log");
    let mut stderr_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .map_err(|error| {
            Error::new(format!(
                "refusing to overwrite worker log {}: {error}",
                stderr_path.display()
            ))
        })?;

    let controller_started_unix_ns = unix_time_ns()?;
    write_json_line(
        &mut output,
        &json!({
            "schema": CONTROLLER_SCHEMA,
            "kind": "controller_start",
            "started_unix_ns": controller_started_unix_ns,
            "source_commit": source_commit,
            "run_id": run_manifest.run_id,
            "session_id": run_manifest.session_id,
            "run_manifest": arguments.run_manifest,
            "run_manifest_sha256": run_manifest_sha256,
            "schedule_sha256": run_manifest.schedule_sha256,
            "model_verification_pre_sha256": run_manifest.model_verification_pre_sha256,
            "model_key": arguments.model.key,
            "model_label": arguments.model.label,
            "context_tokens": arguments.context,
            "generated_tokens": arguments.generated_tokens,
            "warmups": arguments.warmups,
            "trials": arguments.trials,
            "arm": arguments.arm,
            "wired_limit": arguments.wired_limit,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "fixture_rendered_sha256": fixture.rendered_sha256,
            "fixture_token_sha256": fixture.token_sha256,
            "model_manifest_sha256": actual_model_manifest_sha256,
            "oracle_lock_sha256": ORACLE_LOCK_SHA256,
            "worker_sha256": sha256_file(&worker)?,
            "oracle_identity_source_sha256": run_manifest.oracle_identity_source_sha256,
            "oracle_launcher_sha256": run_manifest.oracle_launcher_sha256,
            "model_identity_source_sha256": run_manifest.model_identity_source_sha256,
            "executable_sha256": sha256_file(&executable)?,
            "command": {
                "program": "hyperion-bench",
                "subcommand": "m1 run-cell",
                "output": arguments.output,
            },
            "os_sample_period_ms": SAMPLE_PERIOD.as_millis(),
            "worktree_clean": true,
            "worker_environment": worker_environment(),
            "native_canary": {
                "macos": run_manifest.native_canary.macos,
                "gpu_name": run_manifest.native_canary.gpu_name,
                "mlx_runtime": run_manifest.native_canary.mlx_runtime,
                "recommended_working_set_bytes": run_manifest.native_canary.recommended_working_set_bytes,
            },
        }),
    )?;

    let mut command = Command::new(&python);
    command
        .current_dir(&repo)
        .args(["-I", "-S"])
        .arg(repo.join("oracle/isolated_oracle.py"))
        .arg("script")
        .arg(&worker)
        .args(["--model-path", path_text(&model_path)?])
        .args(["--model-key", arguments.model.key])
        .args(["--model-label", arguments.model.label])
        .args(["--model-manifest-sha256", arguments.model.manifest_sha256])
        .args([
            "--token-file",
            path_text(&corpus_dir.join(&fixture.token_file))?,
        ])
        .args(["--token-sha256", &fixture.token_sha256])
        .args(["--input-tokens", &arguments.context.to_string()])
        .args([
            "--generated-tokens",
            &arguments.generated_tokens.to_string(),
        ])
        .args(["--warmups", &arguments.warmups.to_string()])
        .args(["--trials", &arguments.trials.to_string()])
        .args(["--wired-limit", &arguments.wired_limit])
        .args(["--arm", &arguments.arm])
        .args(["--source-commit", &source_commit])
        .args(["--run-id", &run_manifest.run_id])
        .args(["--session-id", &run_manifest.session_id])
        .args(["--run-manifest-sha256", &run_manifest_sha256])
        .args(["--schedule-sha256", &run_manifest.schedule_sha256])
        .args([
            "--model-verification-receipt-sha256",
            &run_manifest.model_verification_pre_sha256,
        ])
        .args([
            "--expected-recommended-working-set",
            &run_manifest
                .native_canary
                .recommended_working_set_bytes
                .to_string(),
        ])
        .env_clear()
        .envs(worker_environment())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;
    let child_pid = child.id();
    let mut worker_stdin = child
        .stdin
        .take()
        .ok_or_else(|| Error::new("worker stdin pipe was unavailable"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::new("worker stdout pipe was unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::new("worker stderr pipe was unavailable"))?;
    let stderr_thread = thread::spawn(move || -> std::io::Result<Vec<u8>> {
        let mut source = BufReader::new(stderr);
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes)?;
        Ok(bytes)
    });

    let done = Arc::new(AtomicBool::new(false));
    let sample_errors = Arc::new(AtomicUsize::new(0));
    let current_trial = Arc::new(Mutex::new(None::<OpenTrial>));
    let trial_keys = Arc::new(Mutex::new(Vec::<TrialKey>::new()));
    let samples = Arc::new(Mutex::new(Vec::<Sample>::new()));
    let sampler_started = Instant::now();
    let sampler = spawn_sampler(
        child_pid,
        sampler_started,
        Arc::clone(&done),
        Arc::clone(&sample_errors),
        Arc::clone(&current_trial),
        Arc::clone(&samples),
    );

    let mut counts = TraceCounts::default();
    let mut validation_errors = Vec::<String>::new();
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        if line.is_empty() {
            validation_errors.push("worker emitted an empty stdout line".to_owned());
            continue;
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(mut value) => {
                let kind = value.get("kind").and_then(Value::as_str);
                if let Err(error) = observe_worker_value(
                    &value,
                    &arguments,
                    &run_manifest,
                    &run_manifest_sha256,
                    &mut counts,
                    &current_trial,
                    &trial_keys,
                ) {
                    validation_errors.push(error.to_string());
                }
                if kind == Some("trial_start_pending") {
                    sample_boundary(
                        child_pid,
                        sampler_started,
                        SampleBoundary::PreTrial,
                        &sample_errors,
                        &current_trial,
                        &samples,
                    );
                    if let Err(error) =
                        send_worker_command(&mut worker_stdin, "pre_trial_sampled", &value)
                    {
                        validation_errors.push(error.to_string());
                    }
                } else if kind == Some("trial_post_cleanup_pending") {
                    sample_boundary(
                        child_pid,
                        sampler_started,
                        SampleBoundary::PostTrial,
                        &sample_errors,
                        &current_trial,
                        &samples,
                    );
                    if let Err(error) =
                        send_worker_command(&mut worker_stdin, "post_cleanup_sampled", &value)
                    {
                        validation_errors.push(error.to_string());
                    }
                }
                redact_json_paths(&mut value, &repo, &model_path);
                write_json_line(&mut output, &value)?;
            }
            Err(error) => {
                validation_errors.push(format!("non-JSON worker stdout: {error}"));
                write_json_line(
                    &mut output,
                    &json!({
                        "schema": CONTROLLER_SCHEMA,
                        "kind": "invalid_worker_stdout",
                        "line_sha256": sha256_bytes(line.as_bytes()),
                    }),
                )?;
            }
        }
    }
    drop(worker_stdin);
    done.store(true, Ordering::Release);
    sampler
        .join()
        .map_err(|_| Error::new("process-memory sampler panicked"))?;
    let status = child.wait()?;
    let stderr_bytes = stderr_thread
        .join()
        .map_err(|_| Error::new("worker stderr collector panicked"))??;
    let stderr_bytes = redact_bytes_paths(&stderr_bytes, &repo, &model_path);
    stderr_file.write_all(&stderr_bytes)?;
    stderr_file.sync_all()?;

    let trial_keys = trial_keys
        .lock()
        .map_err(|_| Error::new("trial-key lock poisoned"))?;
    let samples = samples
        .lock()
        .map_err(|_| Error::new("sample lock poisoned"))?;
    append_os_samples(&mut output, &samples, &trial_keys)?;
    append_os_summaries(&mut output, &samples, &trial_keys)?;

    let exit = exit_description(status);
    let sample_error_count = sample_errors.load(Ordering::Acquire);
    if sample_error_count > 0 {
        validation_errors.push(format!(
            "proc_pid_rusage failed {sample_error_count} times while the worker was live"
        ));
    }
    if !status.success() {
        validation_errors.push(format!("worker exited unsuccessfully: {exit}"));
    }
    if counts.worker_start != 1
        || counts.worker_end != 1
        || counts.warmups != arguments.warmups
        || counts.measured != arguments.trials
        || counts.failure != 0
        || counts.measured_hashes.len() != 1
        || counts.trial_start_pending != arguments.warmups + arguments.trials
        || counts.trial_start != arguments.warmups + arguments.trials
        || counts.trial_post_cleanup_pending != arguments.warmups + arguments.trials
        || counts.trial_end != arguments.warmups + arguments.trials
    {
        validation_errors.push(format!(
            "worker cardinality mismatch: start={} end={} warmups={} measured={} failures={} hashes={} pending={} running={} post_cleanup={} trial_end={}",
            counts.worker_start,
            counts.worker_end,
            counts.warmups,
            counts.measured,
            counts.failure,
            counts.measured_hashes.len(),
            counts.trial_start_pending,
            counts.trial_start,
            counts.trial_post_cleanup_pending,
            counts.trial_end,
        ));
    }
    if current_trial
        .lock()
        .map_err(|_| Error::new("current-trial lock poisoned"))?
        .is_some()
    {
        validation_errors.push("worker ended with an open trial marker".to_owned());
    }
    for (index, key) in trial_keys.iter().enumerate() {
        if key.phase == "measured" {
            let selected = samples
                .iter()
                .filter(|sample| sample.trial == Some(index))
                .collect::<Vec<_>>();
            if !selected.iter().any(|sample| sample.boundary.is_none()) {
                validation_errors.push(format!(
                    "measured trial {} has no 25 ms periodic OS memory sample",
                    key.trial_index
                ));
            }
            for required in [SampleBoundary::PreTrial, SampleBoundary::PostTrial] {
                if !selected
                    .iter()
                    .any(|sample| sample.boundary == Some(required))
                {
                    validation_errors.push(format!(
                        "measured trial {} lacks the {} OS memory boundary",
                        key.trial_index,
                        required.as_str()
                    ));
                }
            }
        }
    }

    write_json_line(
        &mut output,
        &json!({
            "schema": CONTROLLER_SCHEMA,
            "kind": "controller_end",
            "started_unix_ns": controller_started_unix_ns,
            "finished_unix_ns": unix_time_ns()?,
            "source_commit": source_commit,
            "run_id": run_manifest.run_id,
            "session_id": run_manifest.session_id,
            "run_manifest_sha256": run_manifest_sha256,
            "worker_exit": exit,
            "worker_success": status.success(),
            "uncontrolled_oom": uncontrolled_oom(status, &stderr_bytes, counts.failure),
            "stderr_file": stderr_path.file_name().and_then(OsStr::to_str),
            "stderr_sha256": sha256_bytes(&stderr_bytes),
            "os_samples": samples.len(),
            "os_sample_errors": sample_error_count,
            "validation_errors": validation_errors,
            "valid": validation_errors.is_empty(),
        }),
    )?;
    output.sync_all()?;

    if validation_errors.is_empty() {
        println!(
            "M1_CELL_PASS model={} context={} arm={} output={}",
            arguments.model.key,
            arguments.context,
            arguments.arm,
            arguments.output.display()
        );
        Ok(())
    } else {
        Err(Error::new(format!(
            "M1 cell failed closed: {}",
            validation_errors.join("; ")
        )))
    }
}

fn validate_fixture(
    fixture: &CorpusFixture,
    model: ModelSpec,
    context: u32,
    corpus_dir: &Path,
) -> Result<(), Error> {
    if fixture.model_label != model.label
        || fixture.model_manifest_sha256 != model.manifest_sha256
        || fixture.actual_tokens != context
        || fixture.token_encoding != "little-endian-u32"
        || fixture.tokenizer_sha256
            != "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
        || fixture.template_sha256
            != match model.key {
                "12b" => "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
                "e4b" => "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5",
                _ => return Err(Error::new("unknown M1 model key")),
            }
    {
        return Err(Error::new(
            "M1 corpus fixture identity differs from decision 0002",
        ));
    }
    for name in [&fixture.rendered_file, &fixture.token_file] {
        if !is_plain_filename(name) {
            return Err(Error::new("M1 corpus fixture contains an untrusted path"));
        }
    }
    let rendered_path = corpus_dir.join(&fixture.rendered_file);
    let token_path = corpus_dir.join(&fixture.token_file);
    if sha256_file(&rendered_path)? != fixture.rendered_sha256
        || sha256_file(&token_path)? != fixture.token_sha256
        || fs::metadata(token_path)?.len() != u64::from(context) * 4
    {
        return Err(Error::new(
            "M1 corpus fixture bytes do not match the manifest",
        ));
    }
    Ok(())
}

fn is_plain_filename(value: &str) -> bool {
    let path = Path::new(value);
    !value.is_empty()
        && path.file_name() == Some(OsStr::new(value))
        && path.components().count() == 1
}

fn model_path(repo: &Path, model: ModelSpec) -> PathBuf {
    env::var_os(model.primary_env)
        .or_else(|| model.fallback_env.and_then(env::var_os))
        .map_or_else(|| repo.join(model.default_relative_path), PathBuf::from)
}

fn observe_worker_value(
    value: &Value,
    arguments: &RunCellArgs,
    run_manifest: &RunManifest,
    run_manifest_sha256: &str,
    counts: &mut TraceCounts,
    current_trial: &Mutex<Option<OpenTrial>>,
    trial_keys: &Mutex<Vec<TrialKey>>,
) -> Result<(), Error> {
    let schema = string_field(value, "schema")?;
    let kind = string_field(value, "kind")?;
    match (schema, kind) {
        (WORKER_SCHEMA, "worker_start") => {
            counts.worker_start += 1;
            let requested = u64_field(value, "requested_wired_limit_bytes")?;
            let expected_requested = if arguments.wired_limit == "default" {
                run_manifest.native_canary.recommended_working_set_bytes
            } else {
                arguments
                    .wired_limit
                    .parse::<u64>()
                    .map_err(|_| Error::new("controller wired limit is not u64"))?
            };
            if string_field(value, "model_key")? != arguments.model.key
                || u64_field(value, "input_tokens")? != u64::from(arguments.context)
                || u64_field(value, "generated_tokens")? != u64::from(arguments.generated_tokens)
                || string_field(value, "arm")? != arguments.arm
                || value.get("wired_limit_effective").and_then(Value::as_bool) != Some(true)
                || string_field(value, "source_commit")? != run_manifest.source_commit
                || string_field(value, "run_id")? != run_manifest.run_id
                || string_field(value, "session_id")? != run_manifest.session_id
                || string_field(value, "run_manifest_sha256")? != run_manifest_sha256
                || string_field(value, "schedule_sha256")? != run_manifest.schedule_sha256
                || string_field(value, "model_verification_receipt_sha256")?
                    != run_manifest.model_verification_pre_sha256
                || string_field(value, "python")? != ORACLE_PYTHON
                || string_field(value, "mlx_version")? != MLX_VERSION
                || string_field(value, "mlx_metal_version")? != MLX_METAL_VERSION
                || string_field(value, "mlx_lm_version")? != MLX_LM_VERSION
                || string_field(value, "mlx_lm_commit")? != MLX_LM_COMMIT
                || string_field(value, "mlx_tree_sha256")? != MLX_TREE_SHA256
                || string_field(value, "mlx_metal_tree_sha256")? != MLX_METAL_TREE_SHA256
                || string_field(value, "mlx_lm_tree_sha256")? != MLX_LM_TREE_SHA256
                || u64_field(value, "recommended_working_set_bytes")?
                    != run_manifest.native_canary.recommended_working_set_bytes
                || requested != expected_requested
                || !worker_environment_matches(value.get("environment"))
            {
                return Err(Error::new(
                    "worker-start identity, oracle tree, environment, or run binding does not match controller request",
                ));
            }
            let platform = value
                .get("platform")
                .ok_or_else(|| Error::new("worker-start platform is missing"))?;
            if string_field(platform, "machine")? != "arm64"
                || string_field(platform, "macos")? != run_manifest.native_canary.macos
            {
                return Err(Error::new(
                    "worker platform differs from the run-bound native canary",
                ));
            }
            let _ = u64_field(value, "unix_ns")?;
            let _ = u64_field(value, "monotonic_ns")?;
        }
        (WORKER_SCHEMA, "worker_end") => {
            if current_trial
                .lock()
                .map_err(|_| Error::new("current-trial lock poisoned"))?
                .is_some()
            {
                return Err(Error::new("worker_end occurred with an open trial"));
            }
            counts.worker_end += 1;
        }
        (WORKER_SCHEMA, "failure") => counts.failure += 1,
        (WORKER_SCHEMA, "model_loaded") => {}
        (WORKER_SCHEMA, "trial_start_pending") => {
            let key = trial_key(value)?;
            let mut current = current_trial
                .lock()
                .map_err(|_| Error::new("current-trial lock poisoned"))?;
            if current.is_some() {
                return Err(Error::new("nested worker trial_start marker"));
            }
            let mut keys = trial_keys
                .lock()
                .map_err(|_| Error::new("trial-key lock poisoned"))?;
            keys.push(key);
            *current = Some(OpenTrial {
                index: keys.len() - 1,
                stage: TrialStage::Prepared,
            });
            counts.trial_start_pending += 1;
        }
        (WORKER_SCHEMA, "trial_start") => {
            advance_trial_stage(
                value,
                current_trial,
                trial_keys,
                TrialStage::Prepared,
                TrialStage::Running,
            )?;
            counts.trial_start += 1;
        }
        (WORKER_SCHEMA, "trial_post_cleanup_pending") => {
            advance_trial_stage(
                value,
                current_trial,
                trial_keys,
                TrialStage::Result,
                TrialStage::PostCleanup,
            )?;
            let memory = value
                .get("mlx_memory_post_cleanup")
                .ok_or_else(|| Error::new("post-cleanup MLX memory is missing"))?;
            let _ = u64_field(memory, "active_bytes")?;
            let _ = u64_field(memory, "cache_bytes")?;
            let _ = u64_field(memory, "peak_bytes")?;
            counts.trial_post_cleanup_pending += 1;
        }
        (WORKER_SCHEMA, "trial_end") => {
            let key = trial_key(value)?;
            let mut current = current_trial
                .lock()
                .map_err(|_| Error::new("current-trial lock poisoned"))?;
            let open = current.ok_or_else(|| Error::new("trial_end without trial_start"))?;
            let keys = trial_keys
                .lock()
                .map_err(|_| Error::new("trial-key lock poisoned"))?;
            if keys.get(open.index) != Some(&key) || open.stage != TrialStage::PostCleanup {
                return Err(Error::new(
                    "trial_end does not match a post-cleanup open trial",
                ));
            }
            *current = None;
            counts.trial_end += 1;
        }
        (TRIAL_SCHEMA, "trial") => {
            advance_trial_stage(
                value,
                current_trial,
                trial_keys,
                TrialStage::Running,
                TrialStage::Result,
            )?;
            verify_trial(value, arguments.context, arguments.generated_tokens)?;
            match string_field(value, "phase")? {
                "warmup" => counts.warmups += 1,
                "measured" => {
                    counts.measured += 1;
                    counts
                        .measured_hashes
                        .insert(string_field(value, "output_token_sha256")?.to_owned());
                }
                other => return Err(Error::new(format!("unknown trial phase: {other}"))),
            }
        }
        _ => {
            return Err(Error::new(format!(
                "unsupported worker event schema/kind: {schema}/{kind}"
            )));
        }
    }
    Ok(())
}

fn advance_trial_stage(
    value: &Value,
    current_trial: &Mutex<Option<OpenTrial>>,
    trial_keys: &Mutex<Vec<TrialKey>>,
    expected: TrialStage,
    next: TrialStage,
) -> Result<(), Error> {
    let key = trial_key(value)?;
    let mut current = current_trial
        .lock()
        .map_err(|_| Error::new("current-trial lock poisoned"))?;
    let open = current.ok_or_else(|| Error::new("trial event has no open trial"))?;
    let keys = trial_keys
        .lock()
        .map_err(|_| Error::new("trial-key lock poisoned"))?;
    if open.stage != expected || keys.get(open.index) != Some(&key) {
        return Err(Error::new(format!(
            "trial event has the wrong key or stage: expected {expected:?}, found {:?}",
            open.stage
        )));
    }
    *current = Some(OpenTrial {
        index: open.index,
        stage: next,
    });
    Ok(())
}

fn worker_environment() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("LANG", "C"),
        ("LC_ALL", "C"),
        ("PYTHONHASHSEED", "0"),
        ("TOKENIZERS_PARALLELISM", "false"),
        ("TZ", "UTC"),
    ])
}

fn worker_environment_matches(value: Option<&Value>) -> bool {
    let Some(object) = value.and_then(Value::as_object) else {
        return false;
    };
    let expected = worker_environment();
    object.len() == expected.len()
        && expected.iter().all(|(key, expected_value)| {
            object.get(*key).and_then(Value::as_str) == Some(*expected_value)
        })
}

fn isolated_flags_match(value: Option<&Value>) -> bool {
    let Some(object) = value.and_then(Value::as_object) else {
        return false;
    };
    object.len() == 5
        && object.get("isolated").and_then(Value::as_u64) == Some(1)
        && object.get("no_site").and_then(Value::as_u64) == Some(1)
        && object.get("ignore_environment").and_then(Value::as_u64) == Some(1)
        && object.get("safe_path").and_then(Value::as_bool) == Some(true)
        && object.get("no_user_site").and_then(Value::as_u64) == Some(1)
}

fn send_worker_command(
    worker_stdin: &mut ChildStdin,
    command: &str,
    event: &Value,
) -> Result<(), Error> {
    let message = json!({
        "schema": COMMAND_SCHEMA,
        "command": command,
        "phase": string_field(event, "phase")?,
        "trial_index": u64_field(event, "trial_index")?,
    });
    serde_json::to_writer(&mut *worker_stdin, &message)?;
    worker_stdin.write_all(b"\n")?;
    worker_stdin.flush()?;
    Ok(())
}

fn trial_key(value: &Value) -> Result<TrialKey, Error> {
    Ok(TrialKey {
        phase: string_field(value, "phase")?.to_owned(),
        trial_index: u32::try_from(u64_field(value, "trial_index")?)
            .map_err(|_| Error::new("trial_index exceeds u32"))?,
    })
}

fn verify_trial(value: &Value, input_tokens: u32, generated_tokens: u32) -> Result<(), Error> {
    if u64_field(value, "input_tokens")? != u64::from(input_tokens)
        || u64_field(value, "generated_tokens")? != u64::from(generated_tokens)
    {
        return Err(Error::new("trial input/output cardinality mismatch"));
    }
    let offsets = value
        .get("token_offsets_ns")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("trial token_offsets_ns is not an array"))?
        .iter()
        .map(|item| {
            item.as_u64()
                .ok_or_else(|| Error::new("trial token offset is not u64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if offsets.len() != usize::try_from(generated_tokens).expect("u32 fits usize")
        || offsets.len() < 2
        || offsets.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(Error::new(
            "trial token offsets have invalid cardinality or ordering",
        ));
    }
    let itls = offsets
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect::<Vec<_>>();
    let decode_duration = offsets[offsets.len() - 1] - offsets[0];
    if u64_field(value, "ttft_ns")? != offsets[0]
        || u64_field(value, "decode_duration_ns")? != decode_duration
        || u64_field(value, "decode_intervals")?
            != u64::try_from(itls.len()).expect("interval count fits u64")
        || u64_field(value, "itl_n")? != u64::try_from(itls.len()).expect("interval count fits u64")
        || u64_field(value, "itl_p50_ns")? != nearest_rank(&itls, 50)?
        || u64_field(value, "itl_p95_ns")? != nearest_rank(&itls, 95)?
        || u64_field(value, "itl_p99_ns")? != nearest_rank(&itls, 99)?
    {
        return Err(Error::new(
            "trial timing fields do not recompute from raw offsets",
        ));
    }
    let expected_prefill = f64::from(input_tokens) * 1_000_000_000.0 / offsets[0] as f64;
    let expected_decode = itls.len() as f64 * 1_000_000_000.0 / decode_duration as f64;
    if !approximately_equal(float_field(value, "prefill_tok_s")?, expected_prefill)
        || !approximately_equal(float_field(value, "decode_tok_s")?, expected_decode)
    {
        return Err(Error::new(
            "trial throughput fields do not recompute from raw offsets",
        ));
    }
    if u64_field(value, "mlx_peak_bytes")?
        != value
            .get("mlx_memory_end")
            .and_then(|memory| memory.get("peak_bytes"))
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::new("trial mlx_memory_end.peak_bytes is missing"))?
    {
        return Err(Error::new("trial MLX peak fields disagree"));
    }
    Ok(())
}

fn nearest_rank(values: &[u64], percentile: u32) -> Result<u64, Error> {
    if values.is_empty() || !(1..=100).contains(&percentile) {
        return Err(Error::new("invalid nearest-rank input"));
    }
    let mut ordered = values.to_vec();
    ordered.sort_unstable();
    let rank = (usize::try_from(percentile).expect("percentile fits usize") * ordered.len())
        .div_ceil(100)
        .max(1);
    Ok(ordered[rank - 1])
}

fn approximately_equal(actual: f64, expected: f64) -> bool {
    actual.is_finite()
        && expected.is_finite()
        && (actual - expected).abs() <= (expected.abs() * 1e-12).max(1e-9)
}

fn spawn_sampler(
    pid: u32,
    started: Instant,
    done: Arc<AtomicBool>,
    errors: Arc<AtomicUsize>,
    current_trial: Arc<Mutex<Option<OpenTrial>>>,
    samples: Arc<Mutex<Vec<Sample>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !done.load(Ordering::Acquire) {
            let before = current_trial.lock().ok().and_then(|guard| *guard);
            match hyperion_ffi::sample_process_memory(pid) {
                Ok(memory) => {
                    let elapsed = started.elapsed().as_nanos();
                    let elapsed_ns = u64::try_from(elapsed).unwrap_or(u64::MAX);
                    let after = current_trial.lock().ok().and_then(|guard| *guard);
                    let trial = (before == after)
                        .then_some(before)
                        .flatten()
                        .filter(|open| {
                            matches!(open.stage, TrialStage::Running | TrialStage::Result)
                        })
                        .map(|open| open.index);
                    if let Ok(mut destination) = samples.lock() {
                        destination.push(Sample {
                            elapsed_ns,
                            trial,
                            boundary: None,
                            memory,
                        });
                    } else {
                        errors.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }
                Err(_) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
            thread::sleep(SAMPLE_PERIOD);
        }
    })
}

fn sample_boundary(
    pid: u32,
    started: Instant,
    boundary: SampleBoundary,
    errors: &AtomicUsize,
    current_trial: &Mutex<Option<OpenTrial>>,
    samples: &Mutex<Vec<Sample>>,
) {
    let Ok(memory) = hyperion_ffi::sample_process_memory(pid) else {
        errors.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let Ok(trial) = current_trial
        .lock()
        .map(|guard| guard.map(|open| open.index))
    else {
        errors.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let Ok(mut destination) = samples.lock() else {
        errors.fetch_add(1, Ordering::Relaxed);
        return;
    };
    destination.push(Sample {
        elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        trial,
        boundary: Some(boundary),
        memory,
    });
}

fn append_os_samples(
    output: &mut File,
    samples: &[Sample],
    trial_keys: &[TrialKey],
) -> Result<(), Error> {
    for sample in samples {
        let trial = sample.trial.and_then(|index| trial_keys.get(index));
        write_json_line(
            output,
            &json!({
                "schema": OS_SAMPLE_SCHEMA,
                "kind": "os_sample",
                "elapsed_ns": sample.elapsed_ns,
                "phase": trial.map(|key| key.phase.as_str()),
                "trial_index": trial.map(|key| key.trial_index),
                "boundary": sample.boundary.map(SampleBoundary::as_str),
                "pageins": sample.memory.pageins,
                "wired_size_bytes": sample.memory.wired_size_bytes,
                "resident_size_bytes": sample.memory.resident_size_bytes,
                "phys_footprint_bytes": sample.memory.phys_footprint_bytes,
                "lifetime_max_phys_footprint_bytes": sample
                    .memory
                    .lifetime_max_phys_footprint_bytes,
                "interval_max_phys_footprint_bytes": sample
                    .memory
                    .interval_max_phys_footprint_bytes,
            }),
        )?;
    }
    Ok(())
}

fn append_os_summaries(
    output: &mut File,
    samples: &[Sample],
    trial_keys: &[TrialKey],
) -> Result<(), Error> {
    for (index, key) in trial_keys.iter().enumerate() {
        let mut selected = samples
            .iter()
            .filter(|sample| sample.trial == Some(index))
            .collect::<Vec<_>>();
        selected.sort_by_key(|sample| sample.elapsed_ns);
        let first = selected.first();
        let last = selected.last();
        let pre_trial = selected
            .iter()
            .find(|sample| sample.boundary == Some(SampleBoundary::PreTrial));
        let post_trial = selected
            .iter()
            .find(|sample| sample.boundary == Some(SampleBoundary::PostTrial));
        write_json_line(
            output,
            &json!({
                "schema": OS_SAMPLE_SCHEMA,
                "kind": "os_trial_summary",
                "phase": key.phase,
                "trial_index": key.trial_index,
                "sample_count": selected.len(),
                "pre_trial": pre_trial.map(|sample| memory_json(sample.memory)),
                "post_trial": post_trial.map(|sample| memory_json(sample.memory)),
                "first": first.map(|sample| memory_json(sample.memory)),
                "last": last.map(|sample| memory_json(sample.memory)),
                "max_wired_size_bytes": selected.iter().map(|sample| sample.memory.wired_size_bytes).max(),
                "max_resident_size_bytes": selected.iter().map(|sample| sample.memory.resident_size_bytes).max(),
                "max_phys_footprint_bytes": selected.iter().map(|sample| sample.memory.phys_footprint_bytes).max(),
                "max_interval_phys_footprint_bytes": selected.iter().map(|sample| sample.memory.interval_max_phys_footprint_bytes).max(),
                "max_lifetime_phys_footprint_bytes": selected.iter().map(|sample| sample.memory.lifetime_max_phys_footprint_bytes).max(),
                "pageins_first": first.map(|sample| sample.memory.pageins),
                "pageins_last": last.map(|sample| sample.memory.pageins),
            }),
        )?;
    }
    Ok(())
}

fn memory_json(memory: ProcessMemorySample) -> Value {
    json!({
        "pageins": memory.pageins,
        "wired_size_bytes": memory.wired_size_bytes,
        "resident_size_bytes": memory.resident_size_bytes,
        "phys_footprint_bytes": memory.phys_footprint_bytes,
        "lifetime_max_phys_footprint_bytes": memory.lifetime_max_phys_footprint_bytes,
        "interval_max_phys_footprint_bytes": memory.interval_max_phys_footprint_bytes,
    })
}

fn verify_trace(path: &Path) -> Result<(), Error> {
    let cell = load_cell(path)?;
    if cell.valid {
        println!("M1_TRACE_VALID success {}", path.display());
        return Ok(());
    }
    if cell.failure_recorded && cell.trace_complete && !cell.uncontrolled_oom {
        println!("M1_TRACE_VALID controlled_failure {}", path.display());
        return Ok(());
    }
    Err(Error::new(format!(
        "trace is neither a valid success nor a complete controlled failure: {}",
        cell.validation_errors.join("; ")
    )))
}

fn summarize(directory: &Path) -> Result<(), Error> {
    let cells = load_cells(directory)?;
    let source_commits = cells
        .iter()
        .map(|cell| cell.source_commit.as_str())
        .collect::<BTreeSet<_>>();
    let run_ids = cells
        .iter()
        .map(|cell| cell.run_id.as_str())
        .collect::<BTreeSet<_>>();
    let run_manifests = cells
        .iter()
        .map(|cell| cell.run_manifest_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if source_commits.len() != 1 || run_ids.len() != 1 || run_manifests.len() != 1 {
        return Err(Error::new(
            "M1 summary cannot combine multiple commits, run IDs, or run manifests",
        ));
    }
    let mut unique = BTreeSet::new();
    for cell in &cells {
        if !unique.insert((
            cell.model_key.as_str(),
            cell.context_tokens,
            cell.arm.as_str(),
        )) {
            return Err(Error::new("M1 summary contains a duplicate cell identity"));
        }
    }
    if cells.iter().all(|cell| cell.arm == "core-default") {
        validate_core_matrix(&cells)?;
    } else if cells.iter().all(|cell| cell.arm == "nightly-512x128") {
        let actual = cells
            .iter()
            .map(|cell| {
                (
                    cell.model_key.as_str(),
                    cell.context_tokens,
                    cell.generated_tokens,
                )
            })
            .collect::<BTreeSet<_>>();
        if actual != BTreeSet::from([("12b", 512_u32, 129_u32), ("e4b", 512_u32, 129_u32)])
            || cells.iter().any(|cell| !cell.valid)
        {
            return Err(Error::new(
                "nightly summary requires exactly two valid 12B/E4B 512x128 cells",
            ));
        }
    }
    let summaries = cells
        .iter()
        .map(|cell| -> Result<Value, Error> {
            let metrics = if cell.valid {
                let working_sets = cell
                    .trials
                    .iter()
                    .zip(&cell.os_trials)
                    .map(|(trial, os)| {
                        trial
                            .mlx_active_end_bytes
                            .saturating_add(trial.mlx_cache_end_bytes)
                            .max(os.max_phys_footprint_bytes)
                            .max(os.max_resident_size_bytes) as f64
                    })
                    .collect::<Vec<_>>();
                json!({
                    "prefill_tok_s": distribution(cell.trials.iter().map(|trial| trial.prefill_tok_s))?,
                    "decode_tok_s": distribution(cell.trials.iter().map(|trial| trial.decode_tok_s))?,
                    "ttft_ns": distribution(cell.trials.iter().map(|trial| trial.ttft_ns as f64))?,
                    "itl_p50_ns": distribution(cell.trials.iter().map(|trial| trial.itl_p50_ns as f64))?,
                    "itl_p95_ns": distribution(cell.trials.iter().map(|trial| trial.itl_p95_ns as f64))?,
                    "itl_p99_ns": distribution(cell.trials.iter().map(|trial| trial.itl_p99_ns as f64))?,
                    "mlx_active_end_bytes": distribution(cell.trials.iter().map(|trial| trial.mlx_active_end_bytes as f64))?,
                    "mlx_cache_end_bytes": distribution(cell.trials.iter().map(|trial| trial.mlx_cache_end_bytes as f64))?,
                    "mlx_peak_bytes": distribution(cell.trials.iter().map(|trial| trial.mlx_peak_bytes as f64))?,
                    "os_max_wired_size_bytes": distribution(cell.os_trials.iter().map(|trial| trial.max_wired_size_bytes as f64))?,
                    "os_max_resident_size_bytes": distribution(cell.os_trials.iter().map(|trial| trial.max_resident_size_bytes as f64))?,
                    "os_max_phys_footprint_bytes": distribution(cell.os_trials.iter().map(|trial| trial.max_phys_footprint_bytes as f64))?,
                    "os_max_interval_phys_footprint_bytes": distribution(cell.os_trials.iter().map(|trial| trial.max_interval_phys_footprint_bytes as f64))?,
                    "os_pageins_delta": distribution(cell.os_trials.iter().map(|trial| trial.pageins_last.saturating_sub(trial.pageins_first) as f64))?,
                    "derived_observed_working_set_bytes": distribution(working_sets)?,
                })
            } else {
                Value::Null
            };
            Ok(json!({
                "file": display_raw_path(&cell.path),
                "sha256": cell.sha256,
                "run_id": cell.run_id,
                "session_id": cell.session_id,
                "run_manifest_sha256": cell.run_manifest_sha256,
                "schedule_sha256": cell.schedule_sha256,
                "source_commit": cell.source_commit,
                "model_key": cell.model_key,
                "context_tokens": cell.context_tokens,
                "generated_tokens": cell.generated_tokens,
                "arm": cell.arm,
                "requested_wired_limit_bytes": cell.requested_wired_limit_bytes,
                "recommended_working_set_bytes": cell.recommended_working_set_bytes,
                "started_unix_ns": cell.started_unix_ns,
                "finished_unix_ns": cell.finished_unix_ns,
                "output_token_sha256": cell.output_token_sha256,
                "trial_count": cell.trials.len(),
                "low_n": cell.context_tokens == 131_072 && cell.trials.len() < usize::try_from(TRIALS).expect("u32 fits usize"),
                "gate_eligible": cell.arm != "nightly-512x128" && cell.context_tokens != 131_072,
                "trace_complete": cell.trace_complete,
                "failure_recorded": cell.failure_recorded,
                "uncontrolled_oom": cell.uncontrolled_oom,
                "validation_errors": cell.validation_errors,
                "valid": cell.valid,
                "metrics": metrics,
            }))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let output = json!({
        "schema": "hyperion.m1-summary.v1",
        "input_directory": display_raw_path(directory),
        "source_commits": source_commits,
        "run_ids": run_ids,
        "run_manifest_sha256": run_manifests,
        "cell_count": cells.len(),
        "cells": summaries,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn validate_core_matrix(cells: &[Cell]) -> Result<(), Error> {
    let expected = ["12b", "e4b"]
        .into_iter()
        .flat_map(|model| {
            [512_u32, 1_024, 4_096, 8_192, 16_384, 32_768]
                .into_iter()
                .map(move |context| (model, context))
        })
        .collect::<BTreeSet<_>>();
    let actual = cells
        .iter()
        .map(|cell| (cell.model_key.as_str(), cell.context_tokens))
        .collect::<BTreeSet<_>>();
    if cells.len() != expected.len()
        || actual != expected
        || cells.iter().any(|cell| {
            !cell.valid
                || cell.arm != "core-default"
                || cell.generated_tokens != GENERATED_TOKENS
                || cell.requested_wired_limit_bytes != cell.recommended_working_set_bytes
        })
    {
        return Err(Error::new(
            "core matrix requires exactly twelve valid default-cap 12B/E4B 512/1K/4K/8K/16K/32K cells",
        ));
    }
    Ok(())
}

fn select_budget(directory: &Path) -> Result<(), Error> {
    let cells = load_cells(directory)?
        .into_iter()
        .filter(|cell| cell.arm.starts_with("discovery-"))
        .collect::<Vec<_>>();
    let require_refinement = cells
        .iter()
        .any(|cell| cell.arm.starts_with("discovery-refine-"));
    let result = compute_budget_selection(&cells, require_refinement)?;
    println!("{}", serde_json::to_string_pretty(&result.output)?);
    Ok(())
}

#[derive(Debug)]
struct BudgetResult {
    output: Value,
    selected_c_bytes: u64,
    output_token_sha256: String,
}

#[derive(Debug)]
struct AccaValidation {
    a_cap: u64,
    c_cap: u64,
    a_distribution: Distribution,
    c_distribution: Distribution,
    speedup_proven: bool,
}

fn compute_budget_selection(
    cells: &[Cell],
    require_refinement: bool,
) -> Result<BudgetResult, Error> {
    const GIB: u64 = 1_073_741_824;
    const HALF_GIB: u64 = GIB / 2;
    const COARSE_CAPS: [u64; 8] = [
        4 * GIB,
        5 * GIB,
        6 * GIB,
        7 * GIB,
        8 * GIB,
        9 * GIB,
        10 * GIB,
        11 * GIB,
    ];
    let mut ordered = cells.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|cell| cell.requested_wired_limit_bytes);
    let mut seen = BTreeSet::new();
    let mut commit = None::<&str>;
    let mut run_id = None::<&str>;
    let mut run_manifest = None::<&str>;
    for cell in &ordered {
        if cell.model_key != "12b"
            || cell.context_tokens != 4_096
            || cell.generated_tokens != GENERATED_TOKENS
            || !seen.insert(cell.requested_wired_limit_bytes)
        {
            return Err(Error::new(
                "budget discovery contains a duplicate or non-12B/4K/1025 cell",
            ));
        }
        if let Some(expected) = commit {
            if cell.source_commit != expected {
                return Err(Error::new("budget discovery spans multiple source commits"));
            }
        } else {
            commit = Some(&cell.source_commit);
        }
        if let Some(expected) = run_id {
            if cell.run_id != expected {
                return Err(Error::new("budget discovery spans multiple run IDs"));
            }
        } else {
            run_id = Some(&cell.run_id);
        }
        if let Some(expected) = run_manifest {
            if cell.run_manifest_sha256 != expected {
                return Err(Error::new("budget discovery spans multiple run manifests"));
            }
        } else {
            run_manifest = Some(&cell.run_manifest_sha256);
        }
    }
    let coarse = COARSE_CAPS
        .iter()
        .map(|cap| {
            ordered
                .iter()
                .copied()
                .find(|cell| {
                    cell.requested_wired_limit_bytes == *cap
                        && cell.arm == format!("discovery-coarse-{}g", cap / GIB)
                })
                .ok_or_else(|| Error::new(format!("budget discovery lacks coarse cap {cap}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let eligible_coarse = coarse
        .iter()
        .copied()
        .filter(|cell| budget_cell_eligible(cell))
        .collect::<Vec<_>>();
    if eligible_coarse.is_empty() {
        return Err(Error::new("budget discovery has no eligible coarse points"));
    }
    let coarse_distributions = eligible_coarse
        .iter()
        .map(|cell| distribution(cell.trials.iter().map(|trial| trial.decode_tok_s)))
        .collect::<Result<Vec<_>, _>>()?;
    let coarse_best_index = empirical_best_index(
        &eligible_coarse
            .iter()
            .zip(&coarse_distributions)
            .map(|(cell, values)| (cell.requested_wired_limit_bytes, values.median))
            .collect::<Vec<_>>(),
    )?;
    let coarse_best_cap = eligible_coarse[coarse_best_index].requested_wired_limit_bytes;
    let refinement_candidates = [
        coarse_best_cap.checked_sub(HALF_GIB),
        coarse_best_cap.checked_add(HALF_GIB),
    ]
    .into_iter()
    .flatten()
    .filter(|candidate| (4 * GIB..=11 * GIB).contains(candidate))
    .collect::<Vec<_>>();
    if refinement_candidates.len() != 2 {
        return Err(Error::new(
            "best eligible coarse point is not bracketed by two in-range half-GiB neighbors",
        ));
    }
    let refine = ordered
        .iter()
        .copied()
        .filter(|cell| cell.arm.starts_with("discovery-refine-"))
        .collect::<Vec<_>>();
    if require_refinement {
        if ordered.len() != COARSE_CAPS.len() + refinement_candidates.len()
            || refine.len() != refinement_candidates.len()
        {
            return Err(Error::new(
                "complete budget discovery must contain exactly eight coarse and two refinement outcomes",
            ));
        }
        for cap in &refinement_candidates {
            if !refine.iter().any(|cell| {
                cell.requested_wired_limit_bytes == *cap
                    && cell.arm == format!("discovery-refine-{cap}")
            }) {
                return Err(Error::new(format!(
                    "complete budget discovery lacks refinement cap {cap}"
                )));
            }
        }
    } else if ordered.len() != COARSE_CAPS.len() || !refine.is_empty() {
        return Err(Error::new(
            "coarse budget discovery must contain exactly the eight preregistered outcomes",
        ));
    }

    let selection_cells = if require_refinement {
        ordered.clone()
    } else {
        coarse.clone()
    };
    let eligible = selection_cells
        .iter()
        .copied()
        .filter(|cell| budget_cell_eligible(cell))
        .collect::<Vec<_>>();
    let output_hashes = eligible
        .iter()
        .filter_map(|cell| cell.output_token_sha256.as_deref())
        .collect::<BTreeSet<_>>();
    if output_hashes.len() != 1 {
        return Err(Error::new(
            "eligible budget points do not share one deterministic output hash",
        ));
    }
    let distributions = eligible
        .iter()
        .map(|cell| distribution(cell.trials.iter().map(|trial| trial.decode_tok_s)))
        .collect::<Result<Vec<_>, _>>()?;
    let best_index = empirical_best_index(
        &eligible
            .iter()
            .zip(&distributions)
            .map(|(cell, values)| (cell.requested_wired_limit_bytes, values.median))
            .collect::<Vec<_>>(),
    )?;
    let best = &distributions[best_index];
    let plateau_indices = distributions
        .iter()
        .enumerate()
        .filter_map(|(index, values)| {
            (values.median >= 0.99 * best.median
                && values.max >= best.min
                && best.max >= values.min)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let selected_index = *plateau_indices
        .iter()
        .min_by_key(|index| eligible[**index].requested_wired_limit_bytes)
        .ok_or_else(|| Error::new("budget plateau unexpectedly excluded the empirical best"))?;
    let unique_speed_optimum = distributions
        .iter()
        .enumerate()
        .all(|(index, values)| index == best_index || best.min > values.max);

    let best_cap = eligible[best_index].requested_wired_limit_bytes;
    let points = selection_cells
        .iter()
        .map(|cell| {
            let values = budget_cell_eligible(cell)
                .then(|| distribution(cell.trials.iter().map(|trial| trial.decode_tok_s)))
                .transpose()?;
            Ok(json!({
                "arm": cell.arm,
                "requested_wired_limit_bytes": cell.requested_wired_limit_bytes,
                "decode_tok_s": values,
                "eligible": budget_cell_eligible(cell),
                "outcome": if cell.valid { "success" } else { "failure" },
                "trace_complete": cell.trace_complete,
                "failure_recorded": cell.failure_recorded,
                "uncontrolled_oom": cell.uncontrolled_oom,
                "validation_errors": cell.validation_errors,
                "output_token_sha256": cell.output_token_sha256,
                "file": display_raw_path(&cell.path),
                "sha256": cell.sha256,
            }))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let output = json!({
        "schema": "hyperion.m1-budget-selection.v1",
        "source_commit": commit,
        "run_id": run_id,
        "run_manifest_sha256": run_manifest,
        "schedule_complete": require_refinement,
        "expected_coarse_caps_bytes": COARSE_CAPS,
        "objective": "highest median decode_tok_s",
        "plateau_rule": "median>=99% best and trial ranges overlap",
        "points": points,
        "empirical_best_bytes": best_cap,
        "empirical_best_decode_tok_s": best,
        "unique_speed_optimum": unique_speed_optimum,
        "selection_label": if unique_speed_optimum { "unique" } else { "plateau_or_uncertain" },
        "plateau_bytes": plateau_indices.iter().map(|index| eligible[*index].requested_wired_limit_bytes).collect::<Vec<_>>(),
        "selected_c_bytes": eligible[selected_index].requested_wired_limit_bytes,
        "refinement_candidates_bytes": refinement_candidates,
    });
    Ok(BudgetResult {
        output,
        selected_c_bytes: eligible[selected_index].requested_wired_limit_bytes,
        output_token_sha256: (*output_hashes
            .first()
            .ok_or_else(|| Error::new("budget selection lacks an output hash"))?)
        .to_owned(),
    })
}

fn budget_cell_eligible(cell: &Cell) -> bool {
    cell.valid
        && cell.trace_complete
        && !cell.uncontrolled_oom
        && cell.trials.len() == usize::try_from(TRIALS).expect("u32 fits usize")
        && cell.output_token_sha256.is_some()
}

fn check_acca(directory: &Path) -> Result<(), Error> {
    let output = compute_acca(directory)?;
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn compute_acca(directory: &Path) -> Result<Value, Error> {
    let cells = load_cells(directory)?;
    let mut blocks = BTreeMap::<&str, &Cell>::new();
    for cell in &cells {
        if matches!(
            cell.arm.as_str(),
            "acca-a1" | "acca-c1" | "acca-c2" | "acca-a2"
        ) && blocks.insert(cell.arm.as_str(), cell).is_some()
        {
            return Err(Error::new(format!("duplicate A-C-C-A block: {}", cell.arm)));
        }
    }
    let order = ["acca-a1", "acca-c1", "acca-c2", "acca-a2"];
    if blocks.len() != order.len() || order.iter().any(|arm| !blocks.contains_key(arm)) {
        return Err(Error::new(
            "A-C-C-A evidence requires exactly a1,c1,c2,a2 blocks",
        ));
    }
    let ordered = order.map(|arm| blocks[arm]);
    let first = ordered[0];
    let run_root = directory
        .parent()
        .ok_or_else(|| Error::new("A-C-C-A directory has no run root"))?;
    let selection_path = run_root.join("budget/selection.json");
    let saved_selection: Value = serde_json::from_slice(&fs::read(&selection_path)?)?;
    let budget_cells = load_cells(&run_root.join("budget"))?
        .into_iter()
        .filter(|cell| cell.arm.starts_with("discovery-"))
        .collect::<Vec<_>>();
    let computed_selection = compute_budget_selection(&budget_cells, true)?;
    if saved_selection != computed_selection.output {
        return Err(Error::new(
            "saved budget selection does not recompute from the complete curve",
        ));
    }
    let validation = validate_acca_blocks(ordered, &computed_selection)?;
    let output = json!({
        "schema": "hyperion.m1-acca.v1",
        "source_commit": first.source_commit,
        "run_id": first.run_id,
        "run_manifest_sha256": first.run_manifest_sha256,
        "order": ["A", "C", "C", "A"],
        "chronology_proven": true,
        "selection_file": display_raw_path(&selection_path),
        "selection_sha256": sha256_file(&selection_path)?,
        "output_token_sha256": computed_selection.output_token_sha256,
        "trials_per_block": TRIALS,
        "a_requested_wired_limit_bytes": validation.a_cap,
        "c_requested_wired_limit_bytes": validation.c_cap,
        "a_decode_tok_s": validation.a_distribution,
        "c_decode_tok_s": validation.c_distribution,
        "candidate_min_gt_baseline_max": validation.speedup_proven,
        "speedup_claim_allowed": validation.speedup_proven,
        "operational_default": if validation.speedup_proven { "C" } else { "A" },
        "blocks": order.iter().map(|arm| json!({
            "arm": arm,
            "file": display_raw_path(&blocks[arm].path),
            "sha256": blocks[arm].sha256,
        })).collect::<Vec<_>>(),
    });
    Ok(output)
}

fn validate_acca_blocks(
    ordered: [&Cell; 4],
    selection: &BudgetResult,
) -> Result<AccaValidation, Error> {
    let first = ordered[0];
    for (index, cell) in ordered.iter().enumerate() {
        if !cell.valid
            || cell.model_key != "12b"
            || cell.context_tokens != 4_096
            || cell.source_commit != first.source_commit
            || cell.run_id != first.run_id
            || cell.run_manifest_sha256 != first.run_manifest_sha256
            || cell.generated_tokens != GENERATED_TOKENS
            || cell.recommended_working_set_bytes != first.recommended_working_set_bytes
            || cell.trials.len() != usize::try_from(TRIALS).expect("u32 fits usize")
        {
            return Err(Error::new(
                "A-C-C-A block is invalid, low-N, wrong workload, or source-mismatched",
            ));
        }
        if index > 0 && ordered[index - 1].finished_unix_ns >= cell.started_unix_ns {
            return Err(Error::new(
                "A-C-C-A controller timestamps do not prove chronological A-C-C-A execution",
            ));
        }
    }
    let a_cap = ordered[0].requested_wired_limit_bytes;
    let c_cap = ordered[1].requested_wired_limit_bytes;
    if ordered[3].requested_wired_limit_bytes != a_cap
        || ordered[2].requested_wired_limit_bytes != c_cap
        || a_cap == c_cap
        || a_cap != first.recommended_working_set_bytes
        || c_cap != selection.selected_c_bytes
    {
        return Err(Error::new(
            "A-C-C-A caps are inconsistent, identical, or unbound to the device default and discovery selection",
        ));
    }
    let output_hashes = ordered
        .iter()
        .filter_map(|cell| cell.output_token_sha256.as_deref())
        .collect::<BTreeSet<_>>();
    if output_hashes != BTreeSet::from([selection.output_token_sha256.as_str()]) {
        return Err(Error::new(
            "A-C-C-A and discovery do not share one deterministic output hash",
        ));
    }
    let a_values = [ordered[0], ordered[3]]
        .into_iter()
        .flat_map(|cell| cell.trials.iter().map(|trial| trial.decode_tok_s))
        .collect::<Vec<_>>();
    let c_values = [ordered[1], ordered[2]]
        .into_iter()
        .flat_map(|cell| cell.trials.iter().map(|trial| trial.decode_tok_s))
        .collect::<Vec<_>>();
    let a_distribution = distribution(a_values)?;
    let c_distribution = distribution(c_values)?;
    let speedup_proven = c_distribution.min > a_distribution.max;
    Ok(AccaValidation {
        a_cap,
        c_cap,
        a_distribution,
        c_distribution,
        speedup_proven,
    })
}

fn verify_run(directory: &Path) -> Result<(), Error> {
    let repo = repo_root()?;
    let run_root = if directory.is_absolute() {
        directory.to_owned()
    } else {
        repo.join(directory)
    };
    if !run_root.is_dir() {
        return Err(Error::new(format!(
            "M1 run directory does not exist: {}",
            run_root.display()
        )));
    }
    reject_symlinks(&run_root)?;
    let manifest_path = run_root.join("run-manifest.json");
    let manifest_bytes = fs::read(&manifest_path)?;
    let manifest_value: Value = serde_json::from_slice(&manifest_bytes)?;
    let manifest: RunManifest = serde_json::from_slice(&manifest_bytes)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let executable = env::current_exe()?;
    let mlx_root =
        PathBuf::from(env::var("MLX_ROOT").unwrap_or_else(|_| "/opt/homebrew/opt/mlx".to_owned()));
    let native_mlx_dylib = mlx_root.join("lib/libmlx.dylib");
    let canary_metallib = executable
        .parent()
        .ok_or_else(|| Error::new("benchmark executable has no parent"))?
        .join("hyperion_canary.metallib");
    if manifest.schema != RUN_MANIFEST_SCHEMA
        || run_root.file_name() != Some(OsStr::new(&manifest.run_id))
        || manifest.source_commit != command_stdout(&repo, "git", &["rev-parse", "HEAD"])?
        || manifest.executable_sha256 != sha256_file(&executable)?
        || manifest.native_mlx_dylib_sha256 != sha256_file(&native_mlx_dylib)?
        || manifest.canary_metallib_sha256 != sha256_file(&canary_metallib)?
        || manifest.cargo_lock_sha256 != sha256_file(&repo.join("Cargo.lock"))?
        || manifest.oracle_lock_sha256 != sha256_file(&repo.join("oracle/uv.lock"))?
        || manifest.oracle_lock_sha256 != ORACLE_LOCK_SHA256
        || manifest.corpus_manifest_sha256
            != sha256_file(&repo.join("benchmarks/m1/corpus/manifest.json"))?
        || manifest.schedule_sha256 != sha256_file(&repo.join("benchmarks/m1/schedule.json"))?
        || manifest.worker_sha256 != sha256_file(&repo.join("oracle/m1_bench_worker.py"))?
        || manifest.server_worker_sha256 != sha256_file(&repo.join("oracle/m1_server_smoke.py"))?
        || manifest.oracle_identity_source_sha256
            != sha256_file(&repo.join("oracle/oracle_identity.py"))?
        || manifest.oracle_launcher_sha256 != sha256_file(&repo.join("oracle/isolated_oracle.py"))?
        || manifest.model_identity_source_sha256
            != sha256_file(&repo.join("oracle/model_identity.py"))?
        || manifest.preflight_log_sha256 != sha256_file(&run_root.join("preflight/preflight.log"))?
        || manifest.oracle_verification_pre_sha256
            != sha256_file(&run_root.join("preflight/oracle-verification.log"))?
        || manifest.model_verification_pre_sha256
            != sha256_file(&run_root.join("preflight/model-verification.log"))?
        || manifest.session_id.is_empty()
        || manifest.created_unix_ns == 0
    {
        return Err(Error::new(
            "run manifest does not reproduce against the current clean measurement commit",
        ));
    }
    ensure_clean_worktree(&repo)?;
    let machine_profile = manifest_value
        .get("machine_profile")
        .ok_or_else(|| Error::new("run manifest lacks machine_profile"))?;
    if sha256_bytes(&serde_json::to_vec(machine_profile)?) != manifest.machine_profile_sha256 {
        return Err(Error::new(
            "run manifest machine-profile digest does not recompute",
        ));
    }
    let start_power = machine_profile
        .get("power")
        .and_then(|value| value.get("source"))
        .and_then(Value::as_str);
    let end_power = power_snapshot(&repo);
    if start_power != Some("AC Power")
        || end_power.get("source").and_then(Value::as_str) != Some("AC Power")
    {
        return Err(Error::new(
            "M1 measurement requires AC power at both run boundaries",
        ));
    }
    let end_thermal = thermal_snapshot(&repo);

    let pre_oracle = run_root.join("preflight/oracle-verification.log");
    let post_oracle = run_root.join("postflight/oracle-verification.log");
    let pre_models = run_root.join("preflight/model-verification.log");
    let post_models = run_root.join("postflight/model-verification.log");
    if sha256_file(&pre_oracle)? != sha256_file(&post_oracle)?
        || sha256_file(&pre_models)? != sha256_file(&post_models)?
    {
        return Err(Error::new(
            "pre/post oracle or full-model verification receipts differ",
        ));
    }
    let (expected_oracle, expected_models) = current_verification_receipts(&repo)?;
    if fs::read(&pre_oracle)? != expected_oracle
        || fs::read(&post_oracle)? != expected_oracle
        || fs::read(&pre_models)? != expected_models
        || fs::read(&post_models)? != expected_models
    {
        return Err(Error::new(
            "pre/post oracle or model receipt is not the semantic output of current verification",
        ));
    }

    let core = load_cells(&run_root.join("core"))?;
    validate_core_matrix(&core)?;

    let budget_cells = load_cells(&run_root.join("budget"))?
        .into_iter()
        .filter(|cell| cell.arm.starts_with("discovery-"))
        .collect::<Vec<_>>();
    let coarse_cells = budget_cells
        .iter()
        .filter(|cell| cell.arm.starts_with("discovery-coarse-"))
        .cloned()
        .collect::<Vec<_>>();
    let coarse_budget = compute_budget_selection(&coarse_cells, false)?;
    let saved_coarse: Value =
        serde_json::from_slice(&fs::read(run_root.join("budget/coarse-selection.json"))?)?;
    if saved_coarse != coarse_budget.output {
        return Err(Error::new(
            "saved coarse budget selection does not reproduce from eight outcomes",
        ));
    }
    let budget = compute_budget_selection(&budget_cells, true)?;
    let saved_budget_path = run_root.join("budget/selection.json");
    let saved_budget: Value = serde_json::from_slice(&fs::read(&saved_budget_path)?)?;
    if saved_budget != budget.output || budget_cells.len() != 10 {
        return Err(Error::new(
            "saved complete budget curve does not reproduce from exactly ten outcomes",
        ));
    }
    for cell in &budget_cells {
        if !cell.valid
            && (!cell.failure_recorded
                || cell.uncontrolled_oom
                || cell.finished_unix_ns <= cell.started_unix_ns)
        {
            return Err(Error::new(
                "budget curve contains an unrecorded, uncontrolled, or unterminated failure",
            ));
        }
    }
    let coarse_finish = budget_cells
        .iter()
        .filter(|cell| cell.arm.starts_with("discovery-coarse-"))
        .map(|cell| cell.finished_unix_ns)
        .max()
        .ok_or_else(|| Error::new("budget curve lacks coarse outcomes"))?;
    let refine_start = budget_cells
        .iter()
        .filter(|cell| cell.arm.starts_with("discovery-refine-"))
        .map(|cell| cell.started_unix_ns)
        .min()
        .ok_or_else(|| Error::new("budget curve lacks refinement outcomes"))?;
    if coarse_finish >= refine_start {
        return Err(Error::new(
            "budget timestamps do not prove coarse selection preceded refinement",
        ));
    }

    let acca_path = run_root.join("acca");
    let acca_cells = load_cells(&acca_path)?;
    let acca = compute_acca(&acca_path)?;
    let saved_acca_path = acca_path.join("confirmation.json");
    let saved_acca: Value = serde_json::from_slice(&fs::read(&saved_acca_path)?)?;
    if acca_cells.len() != 4 || saved_acca != acca {
        return Err(Error::new(
            "saved A-C-C-A confirmation does not reproduce from exactly four blocks",
        ));
    }

    let mut all_cells = Vec::new();
    all_cells.extend(core.iter());
    all_cells.extend(budget_cells.iter());
    all_cells.extend(acca_cells.iter());
    if all_cells.iter().any(|cell| {
        cell.source_commit != manifest.source_commit
            || cell.run_id != manifest.run_id
            || cell.session_id != manifest.session_id
            || cell.run_manifest_sha256 != manifest_sha256
            || cell.schedule_sha256 != manifest.schedule_sha256
            || cell.corpus_manifest_sha256 != manifest.corpus_manifest_sha256
            || cell.model_verification_sha256 != manifest.model_verification_pre_sha256
            || cell.worker_sha256 != manifest.worker_sha256
            || cell.executable_sha256 != manifest.executable_sha256
            || cell.recommended_working_set_bytes
                != manifest.native_canary.recommended_working_set_bytes
            || cell.started_unix_ns < manifest.created_unix_ns
            || cell.uncontrolled_oom
    }) {
        return Err(Error::new(
            "M1 cells do not share one commit, run manifest, machine session, or controlled-memory posture",
        ));
    }
    let core_finish = core
        .iter()
        .map(|cell| cell.finished_unix_ns)
        .max()
        .expect("core is nonempty");
    let budget_start = budget_cells
        .iter()
        .map(|cell| cell.started_unix_ns)
        .min()
        .expect("budget is nonempty");
    let budget_finish = budget_cells
        .iter()
        .map(|cell| cell.finished_unix_ns)
        .max()
        .expect("budget is nonempty");
    let acca_start = acca_cells
        .iter()
        .map(|cell| cell.started_unix_ns)
        .min()
        .expect("A-C-C-A is nonempty");
    if core_finish >= budget_start || budget_finish >= acca_start {
        return Err(Error::new(
            "M1 timestamps do not prove core → discovery → confirmation ordering",
        ));
    }

    let mut output_hashes = BTreeMap::<(String, u32, u32), BTreeSet<String>>::new();
    for cell in &all_cells {
        if let Some(hash) = &cell.output_token_sha256 {
            output_hashes
                .entry((
                    cell.model_key.clone(),
                    cell.context_tokens,
                    cell.generated_tokens,
                ))
                .or_default()
                .insert(hash.clone());
        }
    }
    if output_hashes.values().any(|hashes| hashes.len() != 1) {
        return Err(Error::new(
            "same-workload cells do not share deterministic output hashes",
        ));
    }

    let server_12b = verify_server_smoke(&run_root, MODEL_12B, &manifest, &manifest_sha256)?;
    let server_e4b = verify_server_smoke(&run_root, MODEL_E4B, &manifest, &manifest_sha256)?;
    verify_server_file_set(&run_root.join("server"))?;
    let acca_finish = acca_cells
        .iter()
        .map(|cell| cell.finished_unix_ns)
        .max()
        .expect("A-C-C-A is nonempty");
    if acca_finish >= server_12b.0 || server_12b.1 >= server_e4b.0 {
        return Err(Error::new(
            "M1 timestamps do not prove confirmation → 12B server → E4B server ordering",
        ));
    }

    let controlled_budget_failures = budget_cells.iter().filter(|cell| !cell.valid).count();
    let output = json!({
        "schema": "hyperion.m1-run-verification.v1",
        "valid": true,
        "run_id": manifest.run_id,
        "session_id": manifest.session_id,
        "source_commit": manifest.source_commit,
        "run_manifest_sha256": manifest_sha256,
        "executable_sha256": manifest.executable_sha256,
        "native_mlx_dylib_sha256": manifest.native_mlx_dylib_sha256,
        "canary_metallib_sha256": manifest.canary_metallib_sha256,
        "schedule_sha256": manifest.schedule_sha256,
        "machine_profile_sha256": manifest.machine_profile_sha256,
        "recommended_working_set_bytes": manifest.native_canary.recommended_working_set_bytes,
        "core_cells": core.len(),
        "budget_points": budget_cells.len(),
        "controlled_budget_failures": controlled_budget_failures,
        "selected_c_bytes": budget.selected_c_bytes,
        "acca_blocks": acca_cells.len(),
        "server_models": ["12b", "e4b"],
        "output_hash_groups": output_hashes.iter().map(|((model, context, generated), hashes)| json!({
            "model": model,
            "context_tokens": context,
            "generated_tokens": generated,
            "output_token_sha256": hashes.first(),
        })).collect::<Vec<_>>(),
        "pre_post_oracle_sha256": sha256_file(&pre_oracle)?,
        "pre_post_model_verification_sha256": sha256_file(&pre_models)?,
        "end_power": end_power,
        "end_thermal": end_thermal,
        "durable_archive_required": true,
        "durable_archive_kind": "github_release_asset",
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn verify_server_file_set(server_dir: &Path) -> Result<(), Error> {
    let mut expected = BTreeSet::new();
    for model in ["12b", "e4b"] {
        expected.insert(format!("{model}.server-smoke.json"));
        for repeat in 1..=2 {
            expected.insert(format!("{model}-repeat-{repeat}.json"));
            expected.insert(format!("{model}-repeat-{repeat}.server.log"));
            for suffix in ["warmup", "tool-call", "final"] {
                expected.insert(format!("{model}-repeat-{repeat}.{suffix}.jsonl"));
            }
        }
    }
    let actual = fs::read_dir(server_dir)?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(Error::new("server evidence contains a non-file entry"));
            }
            entry
                .file_name()
                .into_string()
                .map_err(|_| Error::new("server evidence filename is not UTF-8"))
        })
        .collect::<Result<BTreeSet<_>, Error>>()?;
    if actual != expected {
        return Err(Error::new(
            "server evidence does not contain exactly the preregistered files",
        ));
    }
    Ok(())
}

fn reject_symlinks(directory: &Path) -> Result<(), Error> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(Error::new(format!(
                "M1 evidence contains a symlink: {}",
                entry.path().display()
            )));
        }
        if file_type.is_dir() {
            reject_symlinks(&entry.path())?;
        }
    }
    Ok(())
}

fn verify_server_smoke(
    run_root: &Path,
    model: ModelSpec,
    manifest: &RunManifest,
    run_manifest_sha256: &str,
) -> Result<(u64, u64), Error> {
    let server_dir = run_root.join("server");
    let model_key = model.key;
    let model_manifest_sha256 = model.manifest_sha256;
    let (model_label, base_port) = match model_key {
        "12b" => (MODEL_12B.label, 18_080_u64),
        "e4b" => (MODEL_E4B.label, 18_090_u64),
        _ => return Err(Error::new("unsupported server-smoke model key")),
    };
    let result_path = server_dir.join(format!("{model_key}.server-smoke.json"));
    if server_dir
        .join(format!("{model_key}.server-smoke.failure.json"))
        .exists()
    {
        return Err(Error::new(format!(
            "{model_key} server smoke retained a failure record"
        )));
    }
    let value: Value = serde_json::from_slice(&fs::read(&result_path)?)?;
    if value.get("schema").and_then(Value::as_str) != Some("hyperion.m1-server-smoke.v1")
        || value.get("success").and_then(Value::as_bool) != Some(true)
        || value.get("source_commit").and_then(Value::as_str) != Some(&manifest.source_commit)
        || value.get("run_id").and_then(Value::as_str) != Some(&manifest.run_id)
        || value.get("run_manifest_sha256").and_then(Value::as_str) != Some(run_manifest_sha256)
        || value.get("model_key").and_then(Value::as_str) != Some(model_key)
        || value.get("model_label").and_then(Value::as_str) != Some(model_label)
        || value.get("model_manifest_sha256").and_then(Value::as_str) != Some(model_manifest_sha256)
        || value
            .get("model_verification_pre_sha256")
            .and_then(Value::as_str)
            != Some(&manifest.model_verification_pre_sha256)
        || value.get("server_source_sha256").and_then(Value::as_str) != Some(MLX_LM_SERVER_SHA256)
        || value.get("worker_sha256").and_then(Value::as_str)
            != Some(&manifest.server_worker_sha256)
        || value.get("oracle_launcher_sha256").and_then(Value::as_str)
            != Some(&manifest.oracle_launcher_sha256)
        || value
            .get("model_identity_source_sha256")
            .and_then(Value::as_str)
            != Some(&manifest.model_identity_source_sha256)
        || value.get("deterministic").and_then(Value::as_bool) != Some(true)
        || value.get("repeat_count").and_then(Value::as_u64) != Some(2)
        || value
            .get("reasoning_validated_empty")
            .and_then(Value::as_bool)
            != Some(true)
        || value.get("usage_validated").and_then(Value::as_bool) != Some(true)
        || value
            .get("finish_reasons_validated")
            .and_then(Value::as_bool)
            != Some(true)
        || value.get("process_exit_validated").and_then(Value::as_bool) != Some(true)
        || value.get("quality_floor").and_then(Value::as_bool) != Some(false)
        || value.get("thinking_enabled").and_then(Value::as_bool) != Some(false)
        || value.get("temperature").and_then(Value::as_f64) != Some(0.0)
        || value.get("seed").and_then(Value::as_u64) != Some(0)
        || value.get("prompt_cache_size").and_then(Value::as_u64) != Some(0)
        || value.get("decode_concurrency").and_then(Value::as_u64) != Some(1)
        || value.get("prompt_concurrency").and_then(Value::as_u64) != Some(1)
        || value.get("prefill_step_size").and_then(Value::as_u64) != Some(2_048)
        || value
            .get("http_inter_emission_is_token_itl")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err(Error::new(format!(
            "{model_key} server smoke envelope is invalid"
        )));
    }
    let identity = value
        .get("oracle_identity")
        .ok_or_else(|| Error::new("server smoke lacks oracle identity"))?;
    if identity.get("python").and_then(Value::as_str) != Some(ORACLE_PYTHON)
        || identity
            .get("python_executable_sha256")
            .and_then(Value::as_str)
            != Some(PYTHON_EXECUTABLE_SHA256)
        || identity
            .get("python_runtime_tree_sha256")
            .and_then(Value::as_str)
            != Some(PYTHON_RUNTIME_TREE_SHA256)
        || identity
            .get("python_runtime_file_count")
            .and_then(Value::as_u64)
            != Some(PYTHON_RUNTIME_FILE_COUNT)
        || identity
            .get("site_packages_tree_sha256")
            .and_then(Value::as_str)
            != Some(SITE_PACKAGES_TREE_SHA256)
        || identity
            .get("site_packages_file_count")
            .and_then(Value::as_u64)
            != Some(SITE_PACKAGES_FILE_COUNT)
        || !isolated_flags_match(identity.get("isolated_flags"))
        || identity.get("mlx_version").and_then(Value::as_str) != Some(MLX_VERSION)
        || identity.get("mlx_metal_version").and_then(Value::as_str) != Some(MLX_METAL_VERSION)
        || identity.get("mlx_lm_version").and_then(Value::as_str) != Some(MLX_LM_VERSION)
        || identity.get("mlx_lm_commit").and_then(Value::as_str) != Some(MLX_LM_COMMIT)
        || identity.get("mlx_tree_sha256").and_then(Value::as_str) != Some(MLX_TREE_SHA256)
        || identity
            .get("mlx_metal_tree_sha256")
            .and_then(Value::as_str)
            != Some(MLX_METAL_TREE_SHA256)
        || identity.get("mlx_lm_tree_sha256").and_then(Value::as_str) != Some(MLX_LM_TREE_SHA256)
    {
        return Err(Error::new(format!(
            "{model_key} server smoke oracle identity is invalid"
        )));
    }
    let initial_model_identity = value
        .get("initial_model_identity")
        .ok_or_else(|| Error::new("server smoke lacks initial exact model identity"))?;
    if !valid_model_identity(initial_model_identity, model) {
        return Err(Error::new(format!(
            "{model_key} server smoke initial model identity is invalid"
        )));
    }
    if !worker_environment_matches(value.get("server_environment")) {
        return Err(Error::new(format!(
            "{model_key} server smoke environment is invalid"
        )));
    }
    let canonical_hashes = value
        .get("canonical_sha256")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("server canonical_sha256 is not an array"))?;
    let repeats = value
        .get("repeats")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("server repeats is not an array"))?;
    if canonical_hashes.len() != 2 || repeats.len() != 2 {
        return Err(Error::new("server smoke does not contain two repeats"));
    }
    for (index, repeat) in repeats.iter().enumerate() {
        let repeat_number = index + 1;
        let port = base_port + u64::try_from(index).expect("repeat index fits u64");
        if repeat.get("repeat").and_then(Value::as_u64)
            != Some(u64::try_from(repeat_number).expect("repeat number fits u64"))
            || repeat.get("port").and_then(Value::as_u64) != Some(port)
            || repeat.get("command") != Some(&expected_server_command(port))
        {
            return Err(Error::new("server repeat command identity is invalid"));
        }
        if repeat.get("model_identity") != Some(initial_model_identity) {
            return Err(Error::new(
                "server repeat did not revalidate the exact model tree before load",
            ));
        }
        let warmup = repeat
            .get("warmup")
            .ok_or_else(|| Error::new("server repeat lacks warmup"))?;
        let first = repeat
            .get("first_turn")
            .ok_or_else(|| Error::new("server repeat lacks first turn"))?;
        let final_turn = repeat
            .get("final_turn")
            .ok_or_else(|| Error::new("server repeat lacks final turn"))?;
        if warmup.get("request") != first.get("request") {
            return Err(Error::new(
                "server warmup and measured first-turn requests differ",
            ));
        }
        if repeat.get("tool_result")
            != Some(&json!([{
                "id": "hq.ahu01.sat",
                "label": "AHU-01 Supply Air Temperature",
                "kind": "analogInput",
                "unit": "degF",
                "tags": ["site:HQ", "equip:AHU-01", "measurement:supply-air-temperature"],
            }]))
        {
            return Err(Error::new("server repeat tool result fixture is invalid"));
        }
        validate_server_turn(warmup, "warmup")?;
        validate_server_turn(first, "first_turn")?;
        validate_server_turn(final_turn, "final_turn")?;
        let canonical = canonical_server_result(first, final_turn)?;
        if repeat.get("canonical") != Some(&canonical) {
            return Err(Error::new(
                "server canonical result does not rebuild from assembled turns",
            ));
        }
        let canonical_sha256 = sha256_bytes(&serde_json::to_vec(&canonical)?);
        if canonical_hashes[index].as_str() != Some(&canonical_sha256) {
            return Err(Error::new("server canonical hash does not recompute"));
        }
        let server_exit = repeat
            .get("server_exit")
            .ok_or_else(|| Error::new("server repeat lacks process exit"))?;
        if !valid_server_exit(server_exit) {
            return Err(Error::new("server repeat has an unexpected process exit"));
        }
        let repeat_path = server_dir.join(format!("{model_key}-repeat-{repeat_number}.json"));
        let repeat_value: Value = serde_json::from_slice(&fs::read(&repeat_path)?)?;
        if &repeat_value != repeat {
            return Err(Error::new(
                "incremental server repeat receipt differs from final result",
            ));
        }
        for (turn, suffix, expected) in [
            ("warmup", "warmup", warmup),
            ("first_turn", "tool-call", first),
            ("final_turn", "final", final_turn),
        ] {
            let name = format!("{model_key}-repeat-{repeat_number}.{suffix}.jsonl");
            let journal_path = server_dir.join(&name);
            let journal = repeat
                .get("journals")
                .and_then(|journals| journals.get(turn))
                .ok_or_else(|| Error::new("server repeat lacks journal receipt"))?;
            if journal.get("file").and_then(Value::as_str) != Some(&name)
                || journal.get("sha256").and_then(Value::as_str)
                    != Some(&sha256_file(&journal_path)?)
            {
                return Err(Error::new(
                    "server journal receipt does not match its bytes",
                ));
            }
            verify_server_journal(&journal_path, expected)?;
        }
        let log_name = format!("{model_key}-repeat-{repeat_number}.server.log");
        let log_path = server_dir.join(&log_name);
        if repeat.get("server_log").and_then(Value::as_str) != Some(&log_name)
            || repeat.get("server_log_sha256").and_then(Value::as_str)
                != Some(&sha256_file(&log_path)?)
        {
            return Err(Error::new(
                "server repeat log receipt does not match its bytes",
            ));
        }
    }
    if canonical_hashes[0] != canonical_hashes[1] {
        return Err(Error::new("server repeats are not canonical-deterministic"));
    }
    let started = u64_field(&value, "started_unix_ns")?;
    let finished = u64_field(&value, "finished_unix_ns")?;
    if finished <= started {
        return Err(Error::new("server smoke timestamps are not increasing"));
    }
    Ok((started, finished))
}

fn valid_usage(value: Option<&Value>) -> bool {
    let Some(value) = value else { return false };
    let Some(prompt) = value.get("prompt_tokens").and_then(Value::as_u64) else {
        return false;
    };
    let Some(completion) = value.get("completion_tokens").and_then(Value::as_u64) else {
        return false;
    };
    let Some(total) = value.get("total_tokens").and_then(Value::as_u64) else {
        return false;
    };
    prompt > 0 && completion > 0 && prompt.checked_add(completion) == Some(total)
}

fn valid_server_exit(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    if object.len() != 3 || value.get("forced_kill").and_then(Value::as_bool) != Some(false) {
        return false;
    }
    match value.get("returncode").and_then(Value::as_i64) {
        Some(0) => value.get("expected_termination").and_then(Value::as_bool) == Some(false),
        Some(-15) => value.get("expected_termination").and_then(Value::as_bool) == Some(true),
        _ => false,
    }
}

fn valid_model_identity(value: &Value, model: ModelSpec) -> bool {
    value.get("schema").and_then(Value::as_str) == Some("hyperion.model-tree-identity.v1")
        && value.get("manifest_sha256").and_then(Value::as_str) == Some(model.manifest_sha256)
        && value.get("payload_tree_sha256").and_then(Value::as_str)
            == Some(model.payload_tree_sha256)
        && value.get("payload_file_count").and_then(Value::as_u64) == Some(model.payload_file_count)
        && value.get("exact_inventory").and_then(Value::as_bool) == Some(true)
        && value.get("symlinks_rejected").and_then(Value::as_bool) == Some(true)
        && value
            .get("transport_cache_excluded")
            .and_then(Value::as_bool)
            == Some(false)
}

fn expected_server_command(port: u64) -> Value {
    json!([
        "oracle/.venv/bin/python",
        "-I",
        "-S",
        "oracle/isolated_oracle.py",
        "module",
        "mlx_lm.server",
        "--model",
        "<MODEL>",
        "--host",
        "127.0.0.1",
        "--port",
        port.to_string(),
        "--temp",
        "0",
        "--max-tokens",
        "128",
        "--chat-template-args",
        "{\"enable_thinking\":false}",
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--prefill-step-size",
        "2048",
        "--prompt-cache-size",
        "0",
    ])
}

fn server_tool_call(assembled: &Value) -> Result<(&Value, Value), Error> {
    let calls = assembled
        .get("tool_calls")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("server assembled tool_calls is not an array"))?;
    if calls.len() != 1 {
        return Err(Error::new(
            "server first turn must contain exactly one tool call",
        ));
    }
    let call = &calls[0];
    let function = call
        .get("function")
        .ok_or_else(|| Error::new("server tool call lacks function"))?;
    let mut arguments = function
        .get("arguments")
        .ok_or_else(|| Error::new("server tool call lacks arguments"))?
        .clone();
    if let Some(encoded) = arguments.as_str() {
        arguments = serde_json::from_str(encoded).map_err(|error| {
            Error::new(format!("server tool arguments are invalid JSON: {error}"))
        })?;
    }
    if call.get("type").and_then(Value::as_str) != Some("function")
        || call.get("index").and_then(Value::as_u64) != Some(0)
        || call
            .get("id")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        || function.get("name").and_then(Value::as_str) != Some("get_points")
        || arguments != json!({"filter": "site:HQ AND equip:AHU-01"})
    {
        return Err(Error::new(
            "server tool call does not match the bounded get_points request",
        ));
    }
    Ok((call, arguments))
}

fn validate_server_turn(turn: &Value, kind: &str) -> Result<(), Error> {
    let request = turn
        .get("request")
        .ok_or_else(|| Error::new(format!("server repeat lacks {kind}.request")))?;
    let expected_messages = if kind == "final_turn" { 3 } else { 1 };
    let tools = request.get("tools").and_then(Value::as_array);
    let messages = request.get("messages").and_then(Value::as_array);
    if turn.get("status").and_then(Value::as_u64) != Some(200)
        || turn
            .get("headers")
            .and_then(|headers| headers.get("content-type"))
            .and_then(Value::as_str)
            .is_none_or(|content_type| !content_type.starts_with("text/event-stream"))
        || request.get("model").and_then(Value::as_str) != Some("default_model")
        || request.get("stream").and_then(Value::as_bool) != Some(true)
        || request
            .get("stream_options")
            .and_then(|options| options.get("include_usage"))
            .and_then(Value::as_bool)
            != Some(true)
        || request.get("temperature").and_then(Value::as_f64) != Some(0.0)
        || request.get("seed").and_then(Value::as_u64) != Some(0)
        || request.get("max_tokens").and_then(Value::as_u64) != Some(128)
        || request
            .get("chat_template_kwargs")
            .and_then(|kwargs| kwargs.get("enable_thinking"))
            .and_then(Value::as_bool)
            != Some(false)
        || tools.is_none_or(|tools| tools.len() != 1)
        || tools
            .and_then(|tools| tools.first())
            .and_then(|tool| tool.get("type"))
            .and_then(Value::as_str)
            != Some("function")
        || tools
            .and_then(|tools| tools.first())
            .and_then(|tool| tool.get("function"))
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            != Some("get_points")
        || messages.is_none_or(|messages| messages.len() != expected_messages)
        || messages
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            != Some("user")
        || messages
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            != Some(
                "Deterministic interoperability check. Call get_points exactly once with filter 'site:HQ AND equip:AHU-01'. Do not answer in prose before the tool result.",
            )
    {
        return Err(Error::new(format!(
            "server {kind} HTTP request or response envelope is invalid"
        )));
    }
    let assembled = turn
        .get("assembled")
        .ok_or_else(|| Error::new(format!("server repeat lacks {kind}.assembled")))?;
    if assembled.get("reasoning_content").and_then(Value::as_str) != Some("")
        || !valid_usage(assembled.get("usage"))
    {
        return Err(Error::new(
            "server repeat reasoning or usage validation does not reproduce",
        ));
    }
    let reasons = assembled
        .get("finish_reasons")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("server finish_reasons is not an array"))?;
    if reasons.len() != 1 {
        return Err(Error::new(
            "server response must contain exactly one finish reason",
        ));
    }
    let terminal = reasons.last().and_then(Value::as_str);
    match kind {
        "warmup" if matches!(terminal, Some("tool_calls" | "stop")) => {}
        "first_turn" if terminal == Some("tool_calls") => {
            let _ = server_tool_call(assembled)?;
        }
        "final_turn"
            if terminal == Some("stop")
                && assembled
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .is_some_and(Vec::is_empty)
                && assembled
                    .get("content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| !content.trim().is_empty()) => {}
        _ => {
            return Err(Error::new(format!(
                "server {kind} terminal shape is invalid"
            )));
        }
    }
    Ok(())
}

fn canonical_server_result(first: &Value, final_turn: &Value) -> Result<Value, Error> {
    let first_assembled = first
        .get("assembled")
        .ok_or_else(|| Error::new("server first turn lacks assembled response"))?;
    let final_assembled = final_turn
        .get("assembled")
        .ok_or_else(|| Error::new("server final turn lacks assembled response"))?;
    let (call, arguments) = server_tool_call(first_assembled)?;
    let final_messages = final_turn
        .get("request")
        .and_then(|request| request.get("messages"))
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new("server final request lacks messages"))?;
    let tool_payload = final_messages
        .get(2)
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new("server final request lacks tool content"))?;
    let parsed_tool_payload: Value = serde_json::from_str(tool_payload).map_err(|error| {
        Error::new(format!(
            "server tool-result content is invalid JSON: {error}"
        ))
    })?;
    if final_messages.first()
        != first
            .get("request")
            .and_then(|request| request.get("messages"))
            .and_then(Value::as_array)
            .and_then(|messages| messages.first())
        || final_messages
            .get(1)
            .and_then(|message| message.get("tool_calls"))
            != Some(&Value::Array(vec![call.clone()]))
        || final_messages
            .get(2)
            .and_then(|message| message.get("tool_call_id"))
            .and_then(Value::as_str)
            != call.get("id").and_then(Value::as_str)
        || final_messages
            .get(2)
            .and_then(|message| message.get("name"))
            .and_then(Value::as_str)
            != Some("get_points")
        || parsed_tool_payload
            != json!([{
                "id": "hq.ahu01.sat",
                "label": "AHU-01 Supply Air Temperature",
                "kind": "analogInput",
                "unit": "degF",
                "tags": ["site:HQ", "equip:AHU-01", "measurement:supply-air-temperature"],
            }])
    {
        return Err(Error::new(
            "server final request is not bound to the first-turn call and tool fixture",
        ));
    }
    Ok(json!({
        "tool_call": {
            "id": "call_1",
            "type": string_field(call, "type")?,
            "function": {
                "name": string_field(call.get("function").ok_or_else(|| Error::new("tool function missing"))?, "name")?,
                "arguments": arguments,
            },
        },
        "first_content": string_field(first_assembled, "content")?,
        "first_reasoning": string_field(first_assembled, "reasoning_content")?,
        "first_finish_reasons": first_assembled.get("finish_reasons").ok_or_else(|| Error::new("first finish reasons missing"))?,
        "first_usage": first_assembled.get("usage").ok_or_else(|| Error::new("first usage missing"))?,
        "final_content": string_field(final_assembled, "content")?,
        "final_reasoning": string_field(final_assembled, "reasoning_content")?,
        "final_finish_reasons": final_assembled.get("finish_reasons").ok_or_else(|| Error::new("final finish reasons missing"))?,
        "final_usage": final_assembled.get("usage").ok_or_else(|| Error::new("final usage missing"))?,
    }))
}

fn assemble_server_chunks(chunks: &[Value]) -> Result<Value, Error> {
    validate_server_chunk_stream(chunks)?;
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut tool_calls = Vec::<Value>::new();
    let mut finish_reasons = Vec::<Value>::new();
    let mut usage = Value::Null;
    for chunk in chunks {
        if let Some(value) = chunk.get("usage").filter(|value| !value.is_null()) {
            usage = value.clone();
        }
        let choices = chunk
            .get("choices")
            .and_then(Value::as_array)
            .expect("chunk schema validation established choices");
        for choice in choices {
            let delta = choice.get("delta").filter(|value| !value.is_null());
            if let Some(delta) = delta {
                if let Some(value) = delta.get("content").filter(|value| !value.is_null()) {
                    content.push_str(
                        value.as_str().ok_or_else(|| {
                            Error::new("server SSE delta.content is not a string")
                        })?,
                    );
                }
                let reasoning_piece = delta
                    .get("reasoning")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .or_else(|| delta.get("reasoning_content").and_then(Value::as_str));
                if let Some(piece) = reasoning_piece {
                    reasoning.push_str(piece);
                }
                if let Some(calls) = delta.get("tool_calls").filter(|value| !value.is_null()) {
                    tool_calls.extend(
                        calls
                            .as_array()
                            .ok_or_else(|| Error::new("server SSE tool_calls is not an array"))?
                            .iter()
                            .cloned(),
                    );
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
                if !reason.is_string() {
                    return Err(Error::new("server SSE finish_reason is not a string"));
                }
                finish_reasons.push(reason.clone());
            }
        }
    }
    Ok(json!({
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }))
}

fn validate_server_chunk_stream(chunks: &[Value]) -> Result<(), Error> {
    if chunks.is_empty() {
        return Err(Error::new("server SSE emitted no JSON chunks"));
    }
    let mut identity = None::<(String, String, String, u64)>;
    let mut usage_indexes = Vec::new();
    let mut finish_reasons = 0_u32;
    for (index, chunk) in chunks.iter().enumerate() {
        if !object_has_only_keys(
            chunk,
            &[
                "id",
                "system_fingerprint",
                "object",
                "model",
                "created",
                "choices",
                "usage",
            ],
        ) || [
            "id",
            "system_fingerprint",
            "object",
            "model",
            "created",
            "choices",
        ]
        .iter()
        .any(|key| chunk.get(*key).is_none())
        {
            return Err(Error::new("server SSE chunk has invalid top-level keys"));
        }
        let current_identity = (
            string_field(chunk, "id")?.to_owned(),
            string_field(chunk, "system_fingerprint")?.to_owned(),
            string_field(chunk, "model")?.to_owned(),
            u64_field(chunk, "created")?,
        );
        if current_identity.0.is_empty()
            || current_identity.1.is_empty()
            || current_identity.2.is_empty()
            || current_identity.3 == 0
            || identity
                .as_ref()
                .is_some_and(|expected| expected != &current_identity)
        {
            return Err(Error::new(
                "server SSE chunk identity is invalid or changed",
            ));
        }
        identity.get_or_insert(current_identity);
        let choices = chunk
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::new("server SSE choices is not an array"))?;
        if choices.len() > 1 {
            return Err(Error::new("server SSE choices has more than one item"));
        }
        if choices.is_empty() {
            let usage = chunk
                .get("usage")
                .ok_or_else(|| Error::new("empty-choice SSE chunk lacks usage"))?;
            if string_field(chunk, "object")? != "chat.completion"
                || !valid_usage(Some(usage))
                || !object_has_only_keys(
                    usage,
                    &[
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                        "prompt_tokens_details",
                    ],
                )
            {
                return Err(Error::new("server SSE usage envelope is invalid"));
            }
            if let Some(details) = usage.get("prompt_tokens_details")
                && (!object_has_only_keys(details, &["cached_tokens"])
                    || details.as_object().is_none_or(|object| object.len() != 1)
                    || details
                        .get("cached_tokens")
                        .and_then(Value::as_u64)
                        .is_none())
            {
                return Err(Error::new("server SSE cached-token details are invalid"));
            }
            usage_indexes.push(index);
            continue;
        }
        if chunk.get("usage").is_some_and(|usage| !usage.is_null())
            || string_field(chunk, "object")? != "chat.completion.chunk"
        {
            return Err(Error::new(
                "nonempty server SSE chunk has usage or the wrong object type",
            ));
        }
        let choice = &choices[0];
        if !object_has_only_keys(choice, &["index", "finish_reason", "delta"])
            || choice.get("index").and_then(Value::as_u64) != Some(0)
        {
            return Err(Error::new("server SSE choice keys or index are invalid"));
        }
        if let Some(reason) = choice.get("finish_reason").filter(|value| !value.is_null()) {
            if !reason.is_string() {
                return Err(Error::new("server SSE finish reason is not a string"));
            }
            finish_reasons += 1;
        }
        let delta = choice
            .get("delta")
            .ok_or_else(|| Error::new("server SSE choice lacks delta"))?;
        if !object_has_only_keys(
            delta,
            &[
                "role",
                "content",
                "reasoning",
                "reasoning_content",
                "tool_calls",
            ],
        ) || delta
            .get("role")
            .is_some_and(|role| role.as_str() != Some("assistant"))
        {
            return Err(Error::new("server SSE delta keys or role are invalid"));
        }
        for field in ["content", "reasoning", "reasoning_content"] {
            if delta.get(field).is_some_and(|value| !value.is_string()) {
                return Err(Error::new(format!(
                    "server SSE delta.{field} is not a string"
                )));
            }
        }
        if delta.get("tool_calls").is_some_and(|calls| {
            calls
                .as_array()
                .is_none_or(|calls| calls.iter().any(|call| !call.is_object()))
        }) {
            return Err(Error::new(
                "server SSE delta.tool_calls is not an array of objects",
            ));
        }
    }
    if usage_indexes != [chunks.len() - 1] || finish_reasons != 1 {
        return Err(Error::new(
            "server SSE requires one finish reason and one terminal usage chunk",
        ));
    }
    Ok(())
}

fn object_has_only_keys(value: &Value, allowed: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object
            .keys()
            .all(|key| allowed.iter().any(|allowed_key| key == allowed_key))
    })
}

fn verify_server_journal(path: &Path, expected_turn: &Value) -> Result<(), Error> {
    let source = File::open(path)?;
    let mut values = Vec::new();
    for (line_index, line) in BufReader::new(source).lines().enumerate() {
        values.push(serde_json::from_str::<Value>(&line?).map_err(|error| {
            Error::new(format!(
                "{} line {} is invalid JSON: {error}",
                path.display(),
                line_index + 1
            ))
        })?);
    }
    if values.len() < 5
        || values
            .first()
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            != Some("request")
        || values
            .get(1)
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            != Some("response_start")
        || values
            .last()
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            != Some("response_end")
    {
        return Err(Error::new(format!(
            "server journal envelope is incomplete: {}",
            path.display()
        )));
    }
    let request = &values[0];
    let response_start = &values[1];
    let response_end = values.last().expect("journal is nonempty");
    let request_unix = u64_field(request, "unix_ns")?;
    let response_start_unix = u64_field(response_start, "unix_ns")?;
    let response_end_unix = u64_field(response_end, "unix_ns")?;
    if !(request_unix <= response_start_unix && response_start_unix <= response_end_unix)
        || response_end.get("saw_done").and_then(Value::as_bool) != Some(true)
    {
        return Err(Error::new(
            "server journal timestamps or terminal marker are invalid",
        ));
    }

    let mut raw_lines = Vec::<Value>::new();
    let mut chunks = Vec::<Value>::new();
    let mut emissions = Vec::<u64>::new();
    let mut saw_done = false;
    for value in &values[2..values.len() - 1] {
        if string_field(value, "kind")? != "sse_line" || saw_done {
            return Err(Error::new(
                "server journal has a non-SSE or post-DONE middle event",
            ));
        }
        let line = string_field(value, "line")?;
        let offset = u64_field(value, "emission_offset_ns")?;
        raw_lines.push(Value::String(line.to_owned()));
        if line.starts_with(':') {
            continue;
        }
        let data = line
            .strip_prefix("data: ")
            .ok_or_else(|| Error::new("server journal contains a malformed SSE line"))?;
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        emissions.push(offset);
        chunks.push(serde_json::from_str(data).map_err(|error| {
            Error::new(format!(
                "server journal SSE payload is invalid JSON: {error}"
            ))
        })?);
    }
    if !saw_done || chunks.is_empty() || emissions.windows(2).any(|pair| pair[1] < pair[0]) {
        return Err(Error::new(
            "server journal lacks ordered SSE chunks and DONE",
        ));
    }
    let inter_emission = emissions
        .windows(2)
        .map(|pair| Value::from(pair[1] - pair[0]))
        .collect::<Vec<_>>();
    let headers = json!({
        "content-type": response_start.get("content_type").cloned().unwrap_or(Value::Null),
        "cache-control": response_start.get("cache_control").cloned().unwrap_or(Value::Null),
    });
    if expected_turn.get("request") != request.get("body")
        || expected_turn.get("status") != response_start.get("status")
        || expected_turn.get("headers") != Some(&headers)
        || expected_turn.get("raw_lines") != Some(&Value::Array(raw_lines))
        || expected_turn.get("chunks") != Some(&Value::Array(chunks.clone()))
        || expected_turn.get("emission_offsets_ns")
            != Some(&Value::Array(
                emissions.iter().copied().map(Value::from).collect(),
            ))
        || expected_turn.get("http_inter_emission_ns") != Some(&Value::Array(inter_emission))
        || expected_turn.get("assembled") != Some(&assemble_server_chunks(&chunks)?)
    {
        return Err(Error::new(
            "server journal does not reproduce its saved request, chunks, timing, or assembly",
        ));
    }
    Ok(())
}

fn load_cells(directory: &Path) -> Result<Vec<Cell>, Error> {
    let repo = repo_root()?;
    let root = if directory.is_absolute() {
        directory.to_owned()
    } else {
        repo.join(directory)
    };
    if !root.is_dir() {
        return Err(Error::new(format!(
            "M1 input directory does not exist: {}",
            root.display()
        )));
    }
    let mut paths = Vec::new();
    collect_jsonl(&root, &mut paths)?;
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new("M1 input directory contains no JSONL cells"));
    }
    paths.iter().map(|path| load_cell(path)).collect()
}

fn collect_jsonl(directory: &Path, output: &mut Vec<PathBuf>) -> Result<(), Error> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(Error::new(format!(
                "M1 evidence directory contains a symlink: {}",
                path.display()
            )));
        }
        if file_type.is_dir() {
            collect_jsonl(&path, output)?;
        } else if path.extension() == Some(OsStr::new("jsonl")) {
            output.push(path);
        }
    }
    Ok(())
}

fn load_cell(path: &Path) -> Result<Cell, Error> {
    let source = File::open(path)?;
    let mut values = Vec::<Value>::new();
    for (line_index, line) in BufReader::new(source).lines().enumerate() {
        let line = line?;
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            Error::new(format!(
                "{} line {} is invalid JSON: {error}",
                path.display(),
                line_index + 1
            ))
        })?;
        values.push(value);
    }
    if values.is_empty() {
        return Err(Error::new(format!("cell {} is empty", path.display())));
    }

    let mut source_commit = None::<String>;
    let mut run_id = None::<String>;
    let mut session_id = None::<String>;
    let mut run_manifest_sha256 = None::<String>;
    let mut schedule_sha256 = None::<String>;
    let mut model_verification_sha256 = None::<String>;
    let mut model_key = None::<String>;
    let mut model_label = None::<String>;
    let mut model_manifest_sha256 = None::<String>;
    let mut corpus_manifest_sha256 = None::<String>;
    let mut fixture_token_sha256 = None::<String>;
    let mut worker_sha256 = None::<String>;
    let mut executable_sha256 = None::<String>;
    let mut oracle_identity_source_sha256 = None::<String>;
    let mut oracle_launcher_sha256 = None::<String>;
    let mut model_identity_source_sha256 = None::<String>;
    let mut native_macos = None::<String>;
    let mut context_tokens = None::<u32>;
    let mut expected_generated_tokens = None::<u32>;
    let mut expected_warmups = None::<u32>;
    let mut expected_trials = None::<u32>;
    let mut arm = None::<String>;
    let mut requested_wired_limit_bytes = None::<u64>;
    let mut recommended_working_set_bytes = None::<u64>;
    let mut started_unix_ns = None::<u64>;
    let mut finished_unix_ns = None::<u64>;
    let mut trials = Vec::<TrialMetrics>::new();
    let mut os_summaries = Vec::<(TrialKey, OsTrialMetrics)>::new();
    let mut os_samples = Vec::<OsSampleEvidence>::new();
    let mut trial_keys = Vec::<TrialKey>::new();
    let mut open_trial = None::<OpenTrial>;
    let mut worker_stage = TraceWorkerStage::AwaitingStart;
    let mut warmups = 0_u32;
    let mut output_hashes = BTreeSet::new();
    let mut protocol_errors = Vec::<String>::new();
    let mut controller_validation_errors = Vec::<String>::new();
    let mut controller_start_count = 0_u32;
    let mut controller_end_count = 0_u32;
    let mut worker_start_count = 0_u32;
    let mut model_loaded_count = 0_u32;
    let mut worker_end_count = 0_u32;
    let mut worker_failure_count = 0_u32;
    let mut worker_capacity_failure_count = 0_u32;
    let mut controller_claimed_valid = false;
    let mut controller_worker_success = false;
    let mut uncontrolled_oom = false;

    for (line_index, value) in values.iter().enumerate() {
        let schema = value.get("schema").and_then(Value::as_str);
        let kind = value.get("kind").and_then(Value::as_str);
        match (schema, kind) {
            (Some(CONTROLLER_SCHEMA), Some("controller_start")) => {
                controller_start_count += 1;
                if line_index != 0 || controller_start_count != 1 {
                    protocol_errors.push("controller_start must be the unique first event".into());
                    continue;
                }
                source_commit = Some(string_field(value, "source_commit")?.to_owned());
                run_id = Some(string_field(value, "run_id")?.to_owned());
                session_id = Some(string_field(value, "session_id")?.to_owned());
                run_manifest_sha256 = Some(string_field(value, "run_manifest_sha256")?.to_owned());
                schedule_sha256 = Some(string_field(value, "schedule_sha256")?.to_owned());
                model_verification_sha256 =
                    Some(string_field(value, "model_verification_pre_sha256")?.to_owned());
                model_key = Some(string_field(value, "model_key")?.to_owned());
                model_label = Some(string_field(value, "model_label")?.to_owned());
                model_manifest_sha256 =
                    Some(string_field(value, "model_manifest_sha256")?.to_owned());
                corpus_manifest_sha256 =
                    Some(string_field(value, "corpus_manifest_sha256")?.to_owned());
                fixture_token_sha256 =
                    Some(string_field(value, "fixture_token_sha256")?.to_owned());
                worker_sha256 = Some(string_field(value, "worker_sha256")?.to_owned());
                executable_sha256 = Some(string_field(value, "executable_sha256")?.to_owned());
                oracle_identity_source_sha256 =
                    Some(string_field(value, "oracle_identity_source_sha256")?.to_owned());
                oracle_launcher_sha256 =
                    Some(string_field(value, "oracle_launcher_sha256")?.to_owned());
                model_identity_source_sha256 =
                    Some(string_field(value, "model_identity_source_sha256")?.to_owned());
                context_tokens = Some(
                    u32::try_from(u64_field(value, "context_tokens")?)
                        .map_err(|_| Error::new("context_tokens exceeds u32"))?,
                );
                expected_generated_tokens = Some(
                    u32::try_from(u64_field(value, "generated_tokens")?)
                        .map_err(|_| Error::new("generated_tokens exceeds u32"))?,
                );
                expected_warmups = Some(
                    u32::try_from(u64_field(value, "warmups")?)
                        .map_err(|_| Error::new("warmups exceeds u32"))?,
                );
                expected_trials = Some(
                    u32::try_from(u64_field(value, "trials")?)
                        .map_err(|_| Error::new("trials exceeds u32"))?,
                );
                arm = Some(string_field(value, "arm")?.to_owned());
                started_unix_ns = Some(u64_field(value, "started_unix_ns")?);
                let recommended =
                    nested_u64_field(value, "native_canary", "recommended_working_set_bytes")?;
                recommended_working_set_bytes = Some(recommended);
                let canary = value
                    .get("native_canary")
                    .ok_or_else(|| Error::new("controller native_canary is missing"))?;
                native_macos = Some(string_field(canary, "macos")?.to_owned());
                let wired = string_field(value, "wired_limit")?;
                requested_wired_limit_bytes = Some(if wired == "default" {
                    recommended
                } else {
                    wired
                        .parse::<u64>()
                        .map_err(|_| Error::new("controller wired_limit is not u64"))?
                });
                let selected_model = match model_key.as_deref() {
                    Some("12b") => MODEL_12B,
                    Some("e4b") => MODEL_E4B,
                    _ => {
                        protocol_errors.push("controller model key is unsupported".into());
                        MODEL_12B
                    }
                };
                let command = value.get("command");
                let run_manifest_path = string_field(value, "run_manifest")?;
                if !is_hex_digest(source_commit.as_deref().unwrap_or_default(), 40)
                    || run_id.as_deref().is_none_or(str::is_empty)
                    || session_id.as_deref().is_none_or(str::is_empty)
                    || !is_hex_digest(run_manifest_sha256.as_deref().unwrap_or_default(), 64)
                    || !is_hex_digest(schedule_sha256.as_deref().unwrap_or_default(), 64)
                    || !is_hex_digest(model_verification_sha256.as_deref().unwrap_or_default(), 64)
                    || model_label.as_deref() != Some(selected_model.label)
                    || model_manifest_sha256.as_deref() != Some(selected_model.manifest_sha256)
                    || !is_hex_digest(corpus_manifest_sha256.as_deref().unwrap_or_default(), 64)
                    || !is_hex_digest(string_field(value, "fixture_rendered_sha256")?, 64)
                    || !is_hex_digest(fixture_token_sha256.as_deref().unwrap_or_default(), 64)
                    || string_field(value, "oracle_lock_sha256")? != ORACLE_LOCK_SHA256
                    || !is_hex_digest(worker_sha256.as_deref().unwrap_or_default(), 64)
                    || !is_hex_digest(
                        oracle_identity_source_sha256.as_deref().unwrap_or_default(),
                        64,
                    )
                    || !is_hex_digest(oracle_launcher_sha256.as_deref().unwrap_or_default(), 64)
                    || !is_hex_digest(
                        model_identity_source_sha256.as_deref().unwrap_or_default(),
                        64,
                    )
                    || !is_hex_digest(executable_sha256.as_deref().unwrap_or_default(), 64)
                    || started_unix_ns == Some(0)
                    || recommended == 0
                    || requested_wired_limit_bytes
                        .is_none_or(|limit| limit == 0 || limit > recommended)
                    || !run_manifest_path.starts_with("benchmarks/raw/m1/")
                    || !run_manifest_path.ends_with("/run-manifest.json")
                    || value.get("os_sample_period_ms").and_then(Value::as_u64) != Some(25)
                    || value.get("worktree_clean").and_then(Value::as_bool) != Some(true)
                    || !worker_environment_matches(value.get("worker_environment"))
                    || string_field(canary, "gpu_name")? != "Apple M5"
                    || string_field(canary, "mlx_runtime")? != MLX_VERSION
                    || !macos_at_least(string_field(canary, "macos")?, 26, 2)
                    || command
                        .and_then(|value| value.get("program"))
                        .and_then(Value::as_str)
                        != Some("hyperion-bench")
                    || command
                        .and_then(|value| value.get("subcommand"))
                        .and_then(Value::as_str)
                        != Some("m1 run-cell")
                    || command
                        .and_then(|value| value.get("output"))
                        .and_then(Value::as_str)
                        .is_none()
                {
                    protocol_errors
                        .push("controller_start provenance or machine envelope is invalid".into());
                }
            }
            (Some(WORKER_SCHEMA), Some("worker_start")) => {
                worker_start_count += 1;
                if worker_stage != TraceWorkerStage::AwaitingStart {
                    protocol_errors.push("worker_start is out of envelope order".into());
                }
                worker_stage = TraceWorkerStage::AwaitingModel;
                let selected_model = match model_key.as_deref() {
                    Some("12b") => MODEL_12B,
                    Some("e4b") => MODEL_E4B,
                    _ => MODEL_12B,
                };
                if worker_start_count != 1
                    || value.get("source_commit").and_then(Value::as_str)
                        != source_commit.as_deref()
                    || value.get("run_id").and_then(Value::as_str) != run_id.as_deref()
                    || value.get("session_id").and_then(Value::as_str) != session_id.as_deref()
                    || value.get("run_manifest_sha256").and_then(Value::as_str)
                        != run_manifest_sha256.as_deref()
                    || value.get("schedule_sha256").and_then(Value::as_str)
                        != schedule_sha256.as_deref()
                    || value
                        .get("model_verification_receipt_sha256")
                        .and_then(Value::as_str)
                        != model_verification_sha256.as_deref()
                    || value.get("model_key").and_then(Value::as_str) != model_key.as_deref()
                    || value.get("model_label").and_then(Value::as_str) != model_label.as_deref()
                    || value.get("model_manifest_sha256").and_then(Value::as_str)
                        != model_manifest_sha256.as_deref()
                    || value.get("model_exact_inventory").and_then(Value::as_bool) != Some(true)
                    || value
                        .get("model_payload_tree_sha256")
                        .and_then(Value::as_str)
                        != Some(selected_model.payload_tree_sha256)
                    || value
                        .get("model_payload_file_count")
                        .and_then(Value::as_u64)
                        != Some(selected_model.payload_file_count)
                    || value.get("token_sha256").and_then(Value::as_str)
                        != fixture_token_sha256.as_deref()
                    || value
                        .get("token_file")
                        .and_then(Value::as_str)
                        .is_none_or(|name| {
                            !is_plain_filename(name) || !name.ends_with(".tokens.u32le")
                        })
                    || value.get("arm").and_then(Value::as_str) != arm.as_deref()
                    || value.get("input_tokens").and_then(Value::as_u64)
                        != context_tokens.map(u64::from)
                    || value.get("generated_tokens").and_then(Value::as_u64)
                        != expected_generated_tokens.map(u64::from)
                    || value.get("warmups").and_then(Value::as_u64)
                        != expected_warmups.map(u64::from)
                    || value.get("trials").and_then(Value::as_u64) != expected_trials.map(u64::from)
                    || value.get("wired_limit_effective").and_then(Value::as_bool) != Some(true)
                    || value.get("python").and_then(Value::as_str) != Some(ORACLE_PYTHON)
                    || value
                        .get("python_executable_sha256")
                        .and_then(Value::as_str)
                        != Some(PYTHON_EXECUTABLE_SHA256)
                    || value
                        .get("python_runtime_tree_sha256")
                        .and_then(Value::as_str)
                        != Some(PYTHON_RUNTIME_TREE_SHA256)
                    || value
                        .get("python_runtime_file_count")
                        .and_then(Value::as_u64)
                        != Some(PYTHON_RUNTIME_FILE_COUNT)
                    || value
                        .get("site_packages_tree_sha256")
                        .and_then(Value::as_str)
                        != Some(SITE_PACKAGES_TREE_SHA256)
                    || value
                        .get("site_packages_file_count")
                        .and_then(Value::as_u64)
                        != Some(SITE_PACKAGES_FILE_COUNT)
                    || !isolated_flags_match(value.get("isolated_flags"))
                    || value.get("mlx_version").and_then(Value::as_str) != Some(MLX_VERSION)
                    || value.get("mlx_metal_version").and_then(Value::as_str)
                        != Some(MLX_METAL_VERSION)
                    || value.get("mlx_lm_version").and_then(Value::as_str) != Some(MLX_LM_VERSION)
                    || value.get("mlx_lm_commit").and_then(Value::as_str) != Some(MLX_LM_COMMIT)
                    || value.get("mlx_tree_sha256").and_then(Value::as_str) != Some(MLX_TREE_SHA256)
                    || value.get("mlx_tree_file_count").and_then(Value::as_u64) != Some(40)
                    || value.get("mlx_metal_tree_sha256").and_then(Value::as_str)
                        != Some(MLX_METAL_TREE_SHA256)
                    || value
                        .get("mlx_metal_tree_file_count")
                        .and_then(Value::as_u64)
                        != Some(406)
                    || value.get("mlx_lm_tree_sha256").and_then(Value::as_str)
                        != Some(MLX_LM_TREE_SHA256)
                    || value.get("mlx_lm_tree_file_count").and_then(Value::as_u64) != Some(176)
                    || value.get("mlx_lm_package_sha256").and_then(Value::as_str)
                        != Some(MLX_LM_PACKAGE_SHA256)
                    || value.get("generate_source_sha256").and_then(Value::as_str)
                        != Some(MLX_LM_GENERATE_SHA256)
                    || value.get("worker_sha256").and_then(Value::as_str)
                        != worker_sha256.as_deref()
                    || value
                        .get("oracle_identity_source_sha256")
                        .and_then(Value::as_str)
                        != oracle_identity_source_sha256.as_deref()
                    || value.get("oracle_launcher_sha256").and_then(Value::as_str)
                        != oracle_launcher_sha256.as_deref()
                    || value
                        .get("model_identity_source_sha256")
                        .and_then(Value::as_str)
                        != model_identity_source_sha256.as_deref()
                    || !worker_environment_matches(value.get("environment"))
                {
                    protocol_errors.push(
                        "worker_start differs from the controller or pinned oracle envelope".into(),
                    );
                }
                if value
                    .get("requested_wired_limit_bytes")
                    .and_then(Value::as_u64)
                    != requested_wired_limit_bytes
                    || value
                        .get("recommended_working_set_bytes")
                        .and_then(Value::as_u64)
                        != recommended_working_set_bytes
                    || value
                        .get("device_info")
                        .and_then(|device| device.get("max_recommended_working_set_size"))
                        .and_then(Value::as_u64)
                        != recommended_working_set_bytes
                {
                    protocol_errors
                        .push("worker wired-limit identity differs from controller".into());
                }
                let platform = value
                    .get("platform")
                    .ok_or_else(|| Error::new("worker platform is missing"))?;
                if platform.get("machine").and_then(Value::as_str) != Some("arm64")
                    || platform.get("macos").and_then(Value::as_str) != native_macos.as_deref()
                    || u64_field(value, "pid")? == 0
                    || u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "monotonic_ns")? == 0
                {
                    protocol_errors.push("worker platform or process envelope is invalid".into());
                }
            }
            (Some(WORKER_SCHEMA), Some("model_loaded")) => {
                model_loaded_count += 1;
                if worker_stage != TraceWorkerStage::AwaitingModel
                    || u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "monotonic_ns")? == 0
                    || u64_field(value, "load_duration_ns")? == 0
                    || validate_mlx_memory(value.get("memory")).is_err()
                {
                    protocol_errors.push("model_loaded envelope is invalid or out of order".into());
                }
                worker_stage = TraceWorkerStage::Trials;
            }
            (Some(WORKER_SCHEMA), Some("trial_start_pending")) => {
                if worker_stage != TraceWorkerStage::Trials {
                    protocol_errors.push("trial_start_pending is outside the trial stage".into());
                }
                let key = trial_key(value)?;
                if u64_field(value, "unix_ns")? == 0 || u64_field(value, "monotonic_ns")? == 0 {
                    protocol_errors.push("trial_start_pending lacks timestamps".into());
                }
                if open_trial.is_some() {
                    protocol_errors.push("nested trial_start_pending".into());
                } else {
                    trial_keys.push(key);
                    open_trial = Some(OpenTrial {
                        index: trial_keys.len() - 1,
                        stage: TrialStage::Prepared,
                    });
                }
            }
            (Some(WORKER_SCHEMA), Some("trial_start")) => {
                if u64_field(value, "unix_ns")? == 0 || u64_field(value, "monotonic_ns")? == 0 {
                    protocol_errors.push("trial_start lacks timestamps".into());
                }
                if let Err(error) = advance_local_trial_stage(
                    value,
                    &mut open_trial,
                    &trial_keys,
                    TrialStage::Prepared,
                    TrialStage::Running,
                ) {
                    protocol_errors.push(error.to_string());
                }
            }
            (Some(TRIAL_SCHEMA), Some("trial")) => {
                if let Err(error) = advance_local_trial_stage(
                    value,
                    &mut open_trial,
                    &trial_keys,
                    TrialStage::Running,
                    TrialStage::Result,
                ) {
                    protocol_errors.push(error.to_string());
                }
                let input = u32::try_from(u64_field(value, "input_tokens")?)
                    .map_err(|_| Error::new("input_tokens exceeds u32"))?;
                let generated = u32::try_from(u64_field(value, "generated_tokens")?)
                    .map_err(|_| Error::new("generated_tokens exceeds u32"))?;
                if Some(input) != context_tokens || Some(generated) != expected_generated_tokens {
                    protocol_errors
                        .push("trial cardinality differs from the controller envelope".into());
                }
                if let Err(error) = verify_trial(value, input, generated) {
                    protocol_errors.push(error.to_string());
                }
                let expected_model_label = match model_key.as_deref() {
                    Some("12b") => MODEL_12B.label,
                    Some("e4b") => MODEL_E4B.label,
                    _ => "",
                };
                if value.get("model_key").and_then(Value::as_str) != model_key.as_deref()
                    || value.get("model_label").and_then(Value::as_str)
                        != Some(expected_model_label)
                    || value.get("arm").and_then(Value::as_str) != arm.as_deref()
                    || value.get("prompt_token_sha256").and_then(Value::as_str)
                        != fixture_token_sha256.as_deref()
                    || value
                        .get("requested_wired_limit_bytes")
                        .and_then(Value::as_u64)
                        != requested_wired_limit_bytes
                    || !is_hex_digest(string_field(value, "output_token_sha256")?, 64)
                    || u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "started_monotonic_ns")?
                        >= u64_field(value, "finished_monotonic_ns")?
                    || validate_mlx_memory(value.get("mlx_memory_start")).is_err()
                    || validate_mlx_memory(value.get("mlx_memory_end")).is_err()
                {
                    protocol_errors
                        .push("trial identity, timing, or memory envelope is invalid".into());
                }
                match string_field(value, "phase")? {
                    "warmup" => warmups += 1,
                    "measured" => {
                        output_hashes
                            .insert(string_field(value, "output_token_sha256")?.to_owned());
                        trials.push(parse_trial_metrics(value)?);
                    }
                    phase => protocol_errors.push(format!("invalid trial phase: {phase}")),
                }
            }
            (Some(WORKER_SCHEMA), Some("trial_post_cleanup_pending")) => {
                if u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "monotonic_ns")? == 0
                    || validate_mlx_memory(value.get("mlx_memory_post_cleanup")).is_err()
                {
                    protocol_errors.push("trial post-cleanup envelope is invalid".into());
                }
                if let Err(error) = advance_local_trial_stage(
                    value,
                    &mut open_trial,
                    &trial_keys,
                    TrialStage::Result,
                    TrialStage::PostCleanup,
                ) {
                    protocol_errors.push(error.to_string());
                }
            }
            (Some(WORKER_SCHEMA), Some("trial_end")) => {
                if u64_field(value, "unix_ns")? == 0 || u64_field(value, "monotonic_ns")? == 0 {
                    protocol_errors.push("trial_end lacks timestamps".into());
                }
                let key = trial_key(value)?;
                match open_trial {
                    Some(open)
                        if open.stage == TrialStage::PostCleanup
                            && trial_keys.get(open.index) == Some(&key) =>
                    {
                        open_trial = None;
                    }
                    _ => protocol_errors.push("trial_end lacks its post-cleanup stage".into()),
                }
            }
            (Some(WORKER_SCHEMA), Some("worker_end")) => {
                worker_end_count += 1;
                if worker_stage != TraceWorkerStage::Trials
                    || open_trial.is_some()
                    || value.get("measured_trials").and_then(Value::as_u64)
                        != expected_trials.map(u64::from)
                    || value.get("warmups").and_then(Value::as_u64)
                        != expected_warmups.map(u64::from)
                    || value
                        .get("measured_output_token_sha256")
                        .and_then(Value::as_str)
                        != output_hashes.first().map(String::as_str)
                    || validate_mlx_memory(value.get("memory")).is_err()
                    || u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "monotonic_ns")? == 0
                {
                    protocol_errors.push("worker_end envelope is invalid or out of order".into());
                }
                worker_stage = TraceWorkerStage::Terminated;
            }
            (Some(WORKER_SCHEMA), Some("failure")) => {
                worker_failure_count += 1;
                let error_type = string_field(value, "error_type")?;
                let failure_message = string_field(value, "message")?;
                let normalized_failure = failure_message.to_ascii_lowercase();
                if error_type == "MemoryError"
                    || [
                        "alloc",
                        "out of memory",
                        "resource exhausted",
                        "resource limit",
                        "wired limit",
                    ]
                    .iter()
                    .any(|needle| normalized_failure.contains(needle))
                {
                    worker_capacity_failure_count += 1;
                }
                if worker_stage == TraceWorkerStage::AwaitingStart
                    || worker_stage == TraceWorkerStage::Terminated
                    || error_type.is_empty()
                    || failure_message.is_empty()
                    || value
                        .get("traceback")
                        .and_then(Value::as_array)
                        .is_none_or(Vec::is_empty)
                    || u64_field(value, "unix_ns")? == 0
                    || u64_field(value, "monotonic_ns")? == 0
                {
                    protocol_errors
                        .push("worker failure envelope is invalid or out of order".into());
                }
                worker_stage = TraceWorkerStage::Terminated;
            }
            (Some(OS_SAMPLE_SCHEMA), Some("os_sample")) => {
                os_samples.push(parse_os_sample(value)?);
            }
            (Some(OS_SAMPLE_SCHEMA), Some("os_trial_summary")) => {
                os_summaries.push((trial_key(value)?, parse_os_trial_metrics(value)?));
            }
            (Some(CONTROLLER_SCHEMA), Some("controller_end")) => {
                controller_end_count += 1;
                if line_index + 1 != values.len() || controller_end_count != 1 {
                    protocol_errors.push("controller_end must be the unique final event".into());
                }
                if value.get("source_commit").and_then(Value::as_str) != source_commit.as_deref()
                    || value.get("run_id").and_then(Value::as_str) != run_id.as_deref()
                    || value.get("session_id").and_then(Value::as_str) != session_id.as_deref()
                    || value.get("run_manifest_sha256").and_then(Value::as_str)
                        != run_manifest_sha256.as_deref()
                {
                    protocol_errors
                        .push("controller_end identity differs from controller_start".into());
                }
                finished_unix_ns = Some(u64_field(value, "finished_unix_ns")?);
                controller_claimed_valid =
                    value.get("valid").and_then(Value::as_bool) == Some(true);
                controller_worker_success =
                    value.get("worker_success").and_then(Value::as_bool) == Some(true);
                uncontrolled_oom =
                    value.get("uncontrolled_oom").and_then(Value::as_bool) == Some(true);
                let errors = value
                    .get("validation_errors")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::new("controller validation_errors is not an array"))?;
                for error in errors {
                    controller_validation_errors.push(
                        error
                            .as_str()
                            .ok_or_else(|| {
                                Error::new("controller validation error is not a string")
                            })?
                            .to_owned(),
                    );
                }
                let worker_exit = value
                    .get("worker_exit")
                    .ok_or_else(|| Error::new("controller worker_exit is missing"))?;
                let stderr_name = string_field(value, "stderr_file")?;
                let code_value = worker_exit.get("code");
                let signal_value = worker_exit.get("signal");
                let exit_fields_valid = code_value
                    .is_some_and(|field| field.is_null() || field.as_i64().is_some())
                    && signal_value
                        .is_some_and(|field| field.is_null() || field.as_i64().is_some());
                let exit_code = code_value.and_then(Value::as_i64);
                let exit_signal = signal_value.and_then(Value::as_i64);
                let derived_worker_success = exit_code == Some(0) && exit_signal.is_none();
                let controlled_failure = exit_code == Some(1)
                    && exit_signal.is_none()
                    && worker_failure_count == 1
                    && worker_capacity_failure_count == 1;
                let strict_exit_shape = derived_worker_success
                    || controlled_failure
                    || (exit_code.is_none() && exit_signal == Some(9));
                let expected_stderr_name = path
                    .with_extension("stderr.log")
                    .file_name()
                    .and_then(OsStr::to_str)
                    .ok_or_else(|| Error::new("cell stderr filename is not UTF-8"))?
                    .to_owned();
                let stderr_bytes = if is_plain_filename(stderr_name) {
                    fs::read(
                        path.parent()
                            .ok_or_else(|| Error::new("cell has no parent directory"))?
                            .join(stderr_name),
                    )?
                } else {
                    Vec::new()
                };
                let derived_uncontrolled_oom =
                    evidence_uncontrolled_oom(exit_signal, &stderr_bytes, worker_failure_count);
                if value.get("started_unix_ns").and_then(Value::as_u64) != started_unix_ns
                    || worker_stage != TraceWorkerStage::Terminated
                    || value.get("os_samples").and_then(Value::as_u64)
                        != Some(u64::try_from(os_samples.len()).expect("sample count fits u64"))
                    || value
                        .get("os_sample_errors")
                        .and_then(Value::as_u64)
                        .is_none()
                    || !is_plain_filename(stderr_name)
                    || !stderr_name.ends_with(".stderr.log")
                    || stderr_name != expected_stderr_name
                    || string_field(value, "stderr_sha256")? != sha256_bytes(&stderr_bytes)
                    || !exit_fields_valid
                    || !strict_exit_shape
                    || controller_worker_success != derived_worker_success
                    || uncontrolled_oom != derived_uncontrolled_oom
                    || (controller_worker_success
                        && (worker_exit.get("code").and_then(Value::as_i64) != Some(0)
                            || !worker_exit.get("signal").is_some_and(Value::is_null)
                            || value.get("os_sample_errors").and_then(Value::as_u64) != Some(0)))
                    || (!controller_worker_success && controller_validation_errors.is_empty())
                    || controller_claimed_valid
                        != (controller_worker_success && controller_validation_errors.is_empty())
                {
                    protocol_errors.push("controller_end execution envelope is invalid".into());
                }
            }
            _ => protocol_errors.push(format!(
                "unsupported event at line {}: {schema:?}/{kind:?}",
                line_index + 1
            )),
        }
    }

    trials.sort_by_key(|trial| trial.trial_index);
    os_summaries.sort_by(|left, right| left.0.cmp(&right.0));
    let started = started_unix_ns.ok_or_else(|| Error::new("cell lacks start timestamp"))?;
    let finished = finished_unix_ns.unwrap_or(0);
    if finished <= started {
        protocol_errors.push("controller timestamps are missing or not increasing".into());
    }
    if open_trial.is_some() && worker_failure_count == 0 {
        protocol_errors.push("trace ended with an open trial".into());
    }
    if worker_start_count != 1 {
        protocol_errors.push(format!(
            "trace requires one worker_start, found {worker_start_count}"
        ));
    }
    if worker_end_count == 1 && worker_failure_count == 0 {
        let expected_keys = (0..expected_warmups.unwrap_or(0))
            .map(|trial_index| TrialKey {
                phase: "warmup".to_owned(),
                trial_index,
            })
            .chain(
                (0..expected_trials.unwrap_or(0)).map(|trial_index| TrialKey {
                    phase: "measured".to_owned(),
                    trial_index,
                }),
            )
            .collect::<Vec<_>>();
        if trial_keys != expected_keys {
            protocol_errors
                .push("worker trial keys are not the exact warmup-then-measured schedule".into());
        }
    }

    let failure_recorded = worker_failure_count > 0 || !controller_worker_success;
    let trace_complete = controller_start_count == 1
        && controller_end_count == 1
        && worker_start_count == 1
        && finished > started
        && protocol_errors.is_empty()
        && ((worker_end_count == 1 && worker_failure_count == 0)
            || (worker_end_count == 0 && worker_failure_count == 1));

    let generated = expected_generated_tokens
        .ok_or_else(|| Error::new("cell lacks generated-token cardinality"))?;
    let context = context_tokens.ok_or_else(|| Error::new("cell lacks context"))?;
    let cell_arm = arm.ok_or_else(|| Error::new("cell lacks arm"))?;
    let standard_cardinality = generated == GENERATED_TOKENS && context != 131_072;
    let nightly_cardinality =
        generated == NIGHTLY_GENERATED_TOKENS && context == 512 && cell_arm == "nightly-512x128";
    let stretch_cardinality = generated == NIGHTLY_GENERATED_TOKENS
        && context == 131_072
        && cell_arm == "stretch-128k"
        && expected_trials.is_some_and(|count| (1..TRIALS).contains(&count));
    let trial_count = u32::try_from(trials.len()).expect("trial count fits u32");
    let required_trials = if stretch_cardinality {
        expected_trials.expect("stretch cardinality has an expected trial count")
    } else {
        TRIALS
    };
    let success_shape = (standard_cardinality || nightly_cardinality || stretch_cardinality)
        && matches!(
            context,
            512 | 1_024 | 4_096 | 8_192 | 16_384 | 32_768 | 131_072
        )
        && matches!(model_key.as_deref(), Some("12b" | "e4b"))
        && expected_warmups == Some(WARMUPS)
        && expected_trials == Some(required_trials)
        && warmups == WARMUPS
        && trial_count == required_trials
        && trials.iter().enumerate().all(|(index, trial)| {
            trial.trial_index == u32::try_from(index).expect("trial index fits u32")
        })
        && output_hashes.len() == 1
        && model_loaded_count == 1
        && worker_end_count == 1
        && worker_failure_count == 0
        && controller_worker_success
        && !uncontrolled_oom;
    let os_errors = if success_shape {
        validate_os_evidence(&os_samples, &os_summaries, &trial_keys)
    } else {
        Vec::new()
    };
    let valid = trace_complete
        && success_shape
        && os_errors.is_empty()
        && controller_claimed_valid
        && controller_validation_errors.is_empty();

    let mut validation_errors = protocol_errors;
    validation_errors.extend(controller_validation_errors);
    validation_errors.extend(os_errors);
    if !success_shape && !failure_recorded {
        validation_errors.push("trace is neither a complete success nor a recorded failure".into());
    }
    let os_trials = os_summaries
        .iter()
        .filter(|(key, _)| key.phase == "measured")
        .map(|(_, summary)| *summary)
        .collect::<Vec<_>>();

    Ok(Cell {
        path: path.to_owned(),
        sha256: sha256_file(path)?,
        run_id: run_id.ok_or_else(|| Error::new("cell lacks run ID"))?,
        session_id: session_id.ok_or_else(|| Error::new("cell lacks session ID"))?,
        run_manifest_sha256: run_manifest_sha256
            .ok_or_else(|| Error::new("cell lacks run-manifest hash"))?,
        schedule_sha256: schedule_sha256.ok_or_else(|| Error::new("cell lacks schedule hash"))?,
        corpus_manifest_sha256: corpus_manifest_sha256
            .ok_or_else(|| Error::new("cell lacks corpus-manifest hash"))?,
        model_verification_sha256: model_verification_sha256
            .ok_or_else(|| Error::new("cell lacks model-verification hash"))?,
        worker_sha256: worker_sha256.ok_or_else(|| Error::new("cell lacks worker hash"))?,
        executable_sha256: executable_sha256
            .ok_or_else(|| Error::new("cell lacks executable hash"))?,
        source_commit: source_commit.ok_or_else(|| Error::new("cell lacks source commit"))?,
        model_key: model_key.ok_or_else(|| Error::new("cell lacks model key"))?,
        context_tokens: context,
        generated_tokens: generated,
        arm: cell_arm,
        requested_wired_limit_bytes: requested_wired_limit_bytes
            .ok_or_else(|| Error::new("cell lacks requested wired limit"))?,
        recommended_working_set_bytes: recommended_working_set_bytes
            .ok_or_else(|| Error::new("cell lacks recommended working set"))?,
        started_unix_ns: started,
        finished_unix_ns: finished,
        output_token_sha256: output_hashes.into_iter().next(),
        trials,
        os_trials,
        validation_errors,
        failure_recorded,
        uncontrolled_oom,
        trace_complete,
        valid,
    })
}

fn advance_local_trial_stage(
    value: &Value,
    current: &mut Option<OpenTrial>,
    trial_keys: &[TrialKey],
    expected: TrialStage,
    next: TrialStage,
) -> Result<(), Error> {
    let key = trial_key(value)?;
    let open = current.ok_or_else(|| Error::new("trial event has no open trial"))?;
    if open.stage != expected || trial_keys.get(open.index) != Some(&key) {
        return Err(Error::new(format!(
            "trial event has the wrong key or stage: expected {expected:?}, found {:?}",
            open.stage
        )));
    }
    *current = Some(OpenTrial {
        index: open.index,
        stage: next,
    });
    Ok(())
}

fn parse_trial_metrics(value: &Value) -> Result<TrialMetrics, Error> {
    Ok(TrialMetrics {
        trial_index: u32::try_from(u64_field(value, "trial_index")?)
            .map_err(|_| Error::new("trial_index exceeds u32"))?,
        prefill_tok_s: float_field(value, "prefill_tok_s")?,
        decode_tok_s: float_field(value, "decode_tok_s")?,
        ttft_ns: u64_field(value, "ttft_ns")?,
        itl_p50_ns: u64_field(value, "itl_p50_ns")?,
        itl_p95_ns: u64_field(value, "itl_p95_ns")?,
        itl_p99_ns: u64_field(value, "itl_p99_ns")?,
        mlx_active_end_bytes: nested_u64_field(value, "mlx_memory_end", "active_bytes")?,
        mlx_cache_end_bytes: nested_u64_field(value, "mlx_memory_end", "cache_bytes")?,
        mlx_peak_bytes: u64_field(value, "mlx_peak_bytes")?,
    })
}

fn parse_os_sample(value: &Value) -> Result<OsSampleEvidence, Error> {
    Ok(OsSampleEvidence {
        elapsed_ns: u64_field(value, "elapsed_ns")?,
        phase: optional_string_field(value, "phase")?,
        trial_index: optional_u32_field(value, "trial_index")?,
        boundary: optional_string_field(value, "boundary")?,
        memory: process_memory_from_json(value)?,
    })
}

fn parse_os_trial_metrics(value: &Value) -> Result<OsTrialMetrics, Error> {
    Ok(OsTrialMetrics {
        trial_index: u32::try_from(u64_field(value, "trial_index")?)
            .map_err(|_| Error::new("OS trial_index exceeds u32"))?,
        sample_count: u64_field(value, "sample_count")?,
        pre_trial: nested_process_memory(value, "pre_trial")?,
        post_trial: nested_process_memory(value, "post_trial")?,
        first: nested_process_memory(value, "first")?,
        last: nested_process_memory(value, "last")?,
        max_wired_size_bytes: u64_field(value, "max_wired_size_bytes")?,
        max_resident_size_bytes: u64_field(value, "max_resident_size_bytes")?,
        max_phys_footprint_bytes: u64_field(value, "max_phys_footprint_bytes")?,
        max_interval_phys_footprint_bytes: u64_field(value, "max_interval_phys_footprint_bytes")?,
        max_lifetime_phys_footprint_bytes: u64_field(value, "max_lifetime_phys_footprint_bytes")?,
        pageins_first: u64_field(value, "pageins_first")?,
        pageins_last: u64_field(value, "pageins_last")?,
    })
}

fn nested_process_memory(value: &Value, field: &str) -> Result<ProcessMemorySample, Error> {
    value
        .get(field)
        .ok_or_else(|| Error::new(format!("OS summary lacks {field}")))
        .and_then(process_memory_from_json)
}

fn process_memory_from_json(value: &Value) -> Result<ProcessMemorySample, Error> {
    Ok(ProcessMemorySample {
        pageins: u64_field(value, "pageins")?,
        wired_size_bytes: u64_field(value, "wired_size_bytes")?,
        resident_size_bytes: u64_field(value, "resident_size_bytes")?,
        phys_footprint_bytes: u64_field(value, "phys_footprint_bytes")?,
        lifetime_max_phys_footprint_bytes: u64_field(value, "lifetime_max_phys_footprint_bytes")?,
        interval_max_phys_footprint_bytes: u64_field(value, "interval_max_phys_footprint_bytes")?,
    })
}

fn validate_os_evidence(
    samples: &[OsSampleEvidence],
    summaries: &[(TrialKey, OsTrialMetrics)],
    trial_keys: &[TrialKey],
) -> Vec<String> {
    let mut errors = Vec::new();
    for sample in samples {
        let identity_shape = matches!(
            (&sample.phase, sample.trial_index),
            (None, None) | (Some(_), Some(_))
        );
        let phase_valid = sample
            .phase
            .as_deref()
            .is_none_or(|phase| matches!(phase, "warmup" | "measured"));
        let boundary_valid = sample.boundary.as_deref().is_none_or(|boundary| {
            matches!(boundary, "pre_trial" | "post_trial") && sample.phase.is_some()
        });
        if sample.elapsed_ns == 0 || !identity_shape || !phase_valid || !boundary_valid {
            errors.push("OS sample has an invalid timestamp, identity, or boundary".into());
        }
    }
    let mut summary_keys = BTreeSet::new();
    for (key, summary) in summaries {
        if summary.trial_index != key.trial_index || !summary_keys.insert(key.clone()) {
            errors.push(format!(
                "duplicate or mismatched OS summary for {} trial {}",
                key.phase, key.trial_index
            ));
        }
    }
    let expected_keys = trial_keys.iter().cloned().collect::<BTreeSet<_>>();
    if summaries.len() != trial_keys.len() || summary_keys != expected_keys {
        errors.push("OS summaries do not exactly cover every warmup and measured trial".into());
    }
    for trial in trial_keys {
        let mut selected = samples
            .iter()
            .filter(|sample| {
                sample.phase.as_deref() == Some(trial.phase.as_str())
                    && sample.trial_index == Some(trial.trial_index)
            })
            .collect::<Vec<_>>();
        selected.sort_by_key(|sample| sample.elapsed_ns);
        let pre = selected
            .iter()
            .filter(|sample| sample.boundary.as_deref() == Some("pre_trial"))
            .collect::<Vec<_>>();
        let post = selected
            .iter()
            .filter(|sample| sample.boundary.as_deref() == Some("post_trial"))
            .collect::<Vec<_>>();
        let periodic = selected
            .iter()
            .filter(|sample| sample.boundary.is_none())
            .collect::<Vec<_>>();
        if pre.len() != 1 || post.len() != 1 || periodic.is_empty() {
            errors.push(format!(
                "{} trial {} needs one pre boundary, one post-cleanup boundary, and periodic samples",
                trial.phase, trial.trial_index
            ));
            continue;
        }
        if !(pre[0].elapsed_ns < periodic[0].elapsed_ns
            && periodic
                .last()
                .is_some_and(|sample| sample.elapsed_ns < post[0].elapsed_ns))
        {
            errors.push(format!(
                "{} trial {} OS boundary ordering is invalid",
                trial.phase, trial.trial_index
            ));
        }
        let Some((_, summary)) = summaries.iter().find(|(key, _)| key == trial) else {
            errors.push(format!(
                "{} trial {} lacks an OS summary",
                trial.phase, trial.trial_index
            ));
            continue;
        };
        let first = selected
            .first()
            .expect("pre boundary makes samples nonempty");
        let last = selected
            .last()
            .expect("post boundary makes samples nonempty");
        let recomputed = (
            u64::try_from(selected.len()).expect("sample count fits u64"),
            selected
                .iter()
                .map(|sample| sample.memory.wired_size_bytes)
                .max()
                .expect("samples are nonempty"),
            selected
                .iter()
                .map(|sample| sample.memory.resident_size_bytes)
                .max()
                .expect("samples are nonempty"),
            selected
                .iter()
                .map(|sample| sample.memory.phys_footprint_bytes)
                .max()
                .expect("samples are nonempty"),
            selected
                .iter()
                .map(|sample| sample.memory.interval_max_phys_footprint_bytes)
                .max()
                .expect("samples are nonempty"),
            selected
                .iter()
                .map(|sample| sample.memory.lifetime_max_phys_footprint_bytes)
                .max()
                .expect("samples are nonempty"),
            first.memory.pageins,
            last.memory.pageins,
        );
        if summary.pre_trial != pre[0].memory
            || summary.post_trial != post[0].memory
            || summary.first != first.memory
            || summary.last != last.memory
            || recomputed
                != (
                    summary.sample_count,
                    summary.max_wired_size_bytes,
                    summary.max_resident_size_bytes,
                    summary.max_phys_footprint_bytes,
                    summary.max_interval_phys_footprint_bytes,
                    summary.max_lifetime_phys_footprint_bytes,
                    summary.pageins_first,
                    summary.pageins_last,
                )
        {
            errors.push(format!(
                "{} trial {} OS summary does not recompute from raw samples",
                trial.phase, trial.trial_index
            ));
        }
    }
    errors
}

fn distribution(values: impl IntoIterator<Item = f64>) -> Result<Distribution, Error> {
    let mut values = values.into_iter().collect::<Vec<_>>();
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::new("distribution requires finite values"));
    }
    values.sort_by(f64::total_cmp);
    let lower = values[(values.len() - 1) / 2];
    let upper = values[values.len() / 2];
    Ok(Distribution {
        min: values[0],
        median: (lower + upper) / 2.0,
        max: values[values.len() - 1],
    })
}

fn empirical_best_index(points: &[(u64, f64)]) -> Result<usize, Error> {
    let mut best = None::<usize>;
    for (index, (cap, median)) in points.iter().enumerate() {
        if !median.is_finite() {
            return Err(Error::new("budget point has a non-finite median"));
        }
        best = match best {
            None => Some(index),
            Some(current)
                if *median > points[current].1
                    || (*median == points[current].1 && *cap < points[current].0) =>
            {
                Some(index)
            }
            Some(current) => Some(current),
        };
    }
    best.ok_or_else(|| Error::new("budget selection has no points"))
}

fn display_raw_path(path: &Path) -> String {
    repo_root()
        .ok()
        .and_then(|repo| path.strip_prefix(repo).ok().map(Path::to_owned))
        .unwrap_or_else(|| path.to_owned())
        .display()
        .to_string()
}

fn ensure_clean_worktree(repo: &Path) -> Result<(), Error> {
    let status = command_stdout(
        repo,
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    if !status.is_empty() {
        return Err(Error::new(format!(
            "M1 measurement requires a clean tracked worktree; found: {}",
            status.replace('\n', "; ")
        )));
    }
    Ok(())
}

fn verify_preflight_receipts(repo: &Path, run_root: &Path) -> Result<(), Error> {
    let (expected_oracle, expected_models) = current_verification_receipts(repo)?;
    if fs::read(run_root.join("preflight/oracle-verification.log"))? != expected_oracle
        || fs::read(run_root.join("preflight/model-verification.log"))? != expected_models
    {
        return Err(Error::new(
            "preflight oracle or model receipt is not the semantic output of current verification",
        ));
    }
    Ok(())
}

fn current_verification_receipts(repo: &Path) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let expected_oracle = command_output(repo, "scripts/verify-oracle.sh", &[])?;
    let mut expected_models = command_output(repo, "scripts/verify-m0-models.sh", &[])?;
    expected_models.extend(command_output(
        repo,
        "scripts/verify-m1-e4b-models.sh",
        &[],
    )?);
    Ok((expected_oracle, expected_models))
}

fn command_output(repo: &Path, program: &str, arguments: &[&str]) -> Result<Vec<u8>, Error> {
    let output = Command::new(program)
        .args(arguments)
        .current_dir(repo)
        .output()?;
    if !output.status.success() {
        return Err(Error::new(format!(
            "{program} {} failed: {}",
            arguments.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn command_stdout(repo: &Path, program: &str, arguments: &[&str]) -> Result<String, Error> {
    Ok(
        String::from_utf8_lossy(&command_output(repo, program, arguments)?)
            .trim()
            .to_owned(),
    )
}

fn repo_root() -> Result<PathBuf, Error> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(Error::from)
}

fn path_text(path: &Path) -> Result<&str, Error> {
    path.to_str()
        .ok_or_else(|| Error::new(format!("path is not UTF-8: {}", path.display())))
}

fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut file = File::open(path)
        .map_err(|error| Error::new(format!("cannot hash {}: {error}", path.display())))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 128 * 1_024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn redact_json_paths(value: &mut Value, repo: &Path, model_path: &Path) {
    match value {
        Value::String(text) => *text = redact_text_paths(text, repo, model_path),
        Value::Array(items) => {
            for item in items {
                redact_json_paths(item, repo, model_path);
            }
        }
        Value::Object(items) => {
            for item in items.values_mut() {
                redact_json_paths(item, repo, model_path);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_bytes_paths(bytes: &[u8], repo: &Path, model_path: &Path) -> Vec<u8> {
    redact_text_paths(&String::from_utf8_lossy(bytes), repo, model_path).into_bytes()
}

fn redact_text_paths(text: &str, repo: &Path, model_path: &Path) -> String {
    let mut redacted = text.to_owned();
    if let Some(model) = model_path.to_str() {
        redacted = redacted.replace(model, "<MODEL>");
    }
    redacted = redact_machine_path(&redacted, repo);
    redacted
}

fn redact_machine_path(text: &str, repo: &Path) -> String {
    let mut redacted = text.to_owned();
    if let Some(repo) = repo.to_str() {
        redacted = redacted.replace(repo, "<REPO>");
    }
    if let Ok(home) = env::var("HOME") {
        redacted = redacted.replace(&home, "<HOME>");
    }
    redacted
}

fn unix_time_ns() -> Result<u64, Error> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::new(format!("system clock precedes Unix epoch: {error}")))?;
    u64::try_from(elapsed.as_nanos()).map_err(|_| Error::new("Unix timestamp exceeds u64"))
}

fn uncontrolled_oom(status: ExitStatus, stderr: &[u8], structured_failures: u32) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        evidence_uncontrolled_oom(status.signal().map(i64::from), stderr, structured_failures)
    }
    #[cfg(not(unix))]
    {
        evidence_uncontrolled_oom(None, stderr, structured_failures)
    }
}

fn evidence_uncontrolled_oom(signal: Option<i64>, stderr: &[u8], structured_failures: u32) -> bool {
    if signal == Some(9) {
        return true;
    }
    if structured_failures > 0 {
        return false;
    }
    let stderr = String::from_utf8_lossy(stderr).to_ascii_lowercase();
    ["out of memory", "std::bad_alloc", "resource exhausted"]
        .iter()
        .any(|needle| stderr.contains(needle))
}

fn write_json_line(output: &mut File, value: &Value) -> Result<(), Error> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    Ok(())
}

fn is_hex_digest(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn macos_at_least(value: &str, required_major: u64, required_minor: u64) -> bool {
    let mut components = value.split('.').map(str::parse::<u64>);
    let (Some(Ok(major)), Some(Ok(minor))) = (components.next(), components.next()) else {
        return false;
    };
    (major, minor) >= (required_major, required_minor)
}

fn validate_mlx_memory(value: Option<&Value>) -> Result<(), Error> {
    let object = value
        .and_then(Value::as_object)
        .ok_or_else(|| Error::new("MLX memory snapshot is not an object"))?;
    if object.len() != 3 {
        return Err(Error::new("MLX memory snapshot has unexpected fields"));
    }
    let read = |field: &str| {
        object
            .get(field)
            .and_then(Value::as_u64)
            .ok_or_else(|| Error::new(format!("MLX memory field {field} is not u64")))
    };
    let active = read("active_bytes")?;
    let _ = read("cache_bytes")?;
    let peak = read("peak_bytes")?;
    if peak < active {
        return Err(Error::new("MLX peak memory is smaller than active memory"));
    }
    Ok(())
}

fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str, Error> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(format!("JSON field {field} is not a string")))
}

fn u64_field(value: &Value, field: &str) -> Result<u64, Error> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::new(format!("JSON field {field} is not u64")))
}

fn optional_string_field(value: &Value, field: &str) -> Result<Option<String>, Error> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(Error::new(format!(
            "JSON field {field} is neither a string nor null"
        ))),
    }
}

fn optional_u32_field(value: &Value, field: &str) -> Result<Option<u32>, Error> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_u64()
            .ok_or_else(|| Error::new(format!("JSON field {field} is not u64")))
            .and_then(|number| {
                u32::try_from(number)
                    .map(Some)
                    .map_err(|_| Error::new(format!("JSON field {field} exceeds u32")))
            }),
        Some(_) => Err(Error::new(format!(
            "JSON field {field} is neither u32 nor null"
        ))),
    }
}

fn nested_u64_field(value: &Value, object: &str, field: &str) -> Result<u64, Error> {
    value
        .get(object)
        .and_then(|nested| nested.get(field))
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::new(format!("JSON field {object}.{field} is not u64")))
}

fn float_field(value: &Value, field: &str) -> Result<f64, Error> {
    value
        .get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| Error::new(format!("JSON field {field} is not a number")))
}

fn exit_description(status: ExitStatus) -> Value {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        json!({"code": status.code(), "signal": status.signal()})
    }
    #[cfg(not(unix))]
    {
        json!({"code": status.code(), "signal": null})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_rank_uses_ceiling_rank() {
        assert_eq!(nearest_rank(&[50, 10, 40, 20, 30], 50).unwrap(), 30);
        assert_eq!(nearest_rank(&[50, 10, 40, 20, 30], 95).unwrap(), 50);
        assert_eq!(nearest_rank(&[50, 10, 40, 20, 30], 99).unwrap(), 50);
    }

    #[test]
    fn rejects_non_raw_or_parent_output_paths() {
        for invalid in [
            "/tmp/result.jsonl",
            "benchmarks/result.jsonl",
            "benchmarks/raw/m1/../escape.jsonl",
            "benchmarks/raw/m1/result.txt",
        ] {
            assert!(
                validate_raw_output_path(invalid).is_err(),
                "accepted {invalid}"
            );
        }
        assert!(validate_raw_output_path("benchmarks/raw/m1/12b-1k.jsonl").is_ok());
    }

    #[test]
    fn parses_only_preregistered_scored_cardinality() {
        let valid = [
            "--model",
            "12b",
            "--context",
            "4096",
            "--arm",
            "core",
            "--wired-limit",
            "default",
            "--output",
            "benchmarks/raw/m1/test.jsonl",
            "--run-manifest",
            "benchmarks/raw/m1/test/run-manifest.json",
        ]
        .map(str::to_owned);
        let parsed = parse_run_cell(&valid).unwrap();
        assert_eq!(parsed.generated_tokens, 1_025);
        assert_eq!(parsed.warmups, 1);
        assert_eq!(parsed.trials, 5);

        let mut invalid = valid.to_vec();
        invalid.extend(["--trials".to_owned(), "4".to_owned()]);
        assert!(parse_run_cell(&invalid).is_err());

        let nightly = [
            "--model",
            "e4b",
            "--context",
            "512",
            "--arm",
            "nightly-512x128",
            "--wired-limit",
            "default",
            "--generated-tokens",
            "129",
            "--output",
            "benchmarks/raw/m1/nightly.jsonl",
            "--run-manifest",
            "benchmarks/raw/m1/nightly/run-manifest.json",
        ]
        .map(str::to_owned);
        assert_eq!(
            parse_run_cell(&nightly).unwrap().generated_tokens,
            NIGHTLY_GENERATED_TOKENS
        );

        let stretch = [
            "--model",
            "e4b",
            "--context",
            "131072",
            "--arm",
            "stretch-128k",
            "--wired-limit",
            "default",
            "--generated-tokens",
            "129",
            "--trials",
            "2",
            "--output",
            "benchmarks/raw/m1/stretch/e4b.jsonl",
            "--run-manifest",
            "benchmarks/raw/m1/stretch/run-manifest.json",
        ]
        .map(str::to_owned);
        assert_eq!(parse_run_cell(&stretch).unwrap().trials, 2);
    }

    #[test]
    fn recomputes_trial_boundaries_and_percentiles() {
        let offsets = [100_u64, 110, 130, 160, 200, 250];
        let itls = [10_u64, 20, 30, 40, 50];
        let value = json!({
            "input_tokens": 4,
            "generated_tokens": 6,
            "decode_intervals": 5,
            "token_offsets_ns": offsets,
            "ttft_ns": 100,
            "prefill_tok_s": 40_000_000.0,
            "decode_duration_ns": 150,
            "decode_tok_s": 33_333_333.333333332,
            "itl_n": 5,
            "itl_p50_ns": nearest_rank(&itls, 50).unwrap(),
            "itl_p95_ns": nearest_rank(&itls, 95).unwrap(),
            "itl_p99_ns": nearest_rank(&itls, 99).unwrap(),
            "mlx_peak_bytes": 123,
            "mlx_memory_end": {"peak_bytes": 123},
        });
        verify_trial(&value, 4, 6).unwrap();
    }

    #[test]
    fn distribution_uses_conventional_even_median() {
        let values = distribution([1.0, 9.0, 3.0, 5.0]).unwrap();
        assert_eq!(values.min, 1.0);
        assert_eq!(values.median, 4.0);
        assert_eq!(values.max, 9.0);
    }

    #[test]
    fn empirical_budget_tie_prefers_smaller_cap() {
        let points = [(8_u64, 3.0), (6, 3.0), (4, 2.0)];
        assert_eq!(empirical_best_index(&points).unwrap(), 1);
    }

    fn synthetic_cell(arm: &str, cap: u64, median: f64, started: u64) -> Cell {
        let values = [median - 1.0, median, median, median, median + 1.0];
        let trials = values
            .into_iter()
            .enumerate()
            .map(|(index, decode_tok_s)| TrialMetrics {
                trial_index: u32::try_from(index).unwrap(),
                prefill_tok_s: 1.0,
                decode_tok_s,
                ttft_ns: 1,
                itl_p50_ns: 1,
                itl_p95_ns: 1,
                itl_p99_ns: 1,
                mlx_active_end_bytes: 1,
                mlx_cache_end_bytes: 1,
                mlx_peak_bytes: 1,
            })
            .collect::<Vec<_>>();
        let memory = ProcessMemorySample {
            pageins: 0,
            wired_size_bytes: 1,
            resident_size_bytes: 1,
            phys_footprint_bytes: 1,
            lifetime_max_phys_footprint_bytes: 1,
            interval_max_phys_footprint_bytes: 1,
        };
        let os_trials = (0..5)
            .map(|trial_index| OsTrialMetrics {
                trial_index,
                sample_count: 3,
                pre_trial: memory,
                post_trial: memory,
                first: memory,
                last: memory,
                max_wired_size_bytes: 1,
                max_resident_size_bytes: 1,
                max_phys_footprint_bytes: 1,
                max_interval_phys_footprint_bytes: 1,
                max_lifetime_phys_footprint_bytes: 1,
                pageins_first: 0,
                pageins_last: 0,
            })
            .collect();
        Cell {
            path: PathBuf::from(format!("{arm}.jsonl")),
            sha256: "1".repeat(64),
            run_id: "run".into(),
            session_id: "session".into(),
            run_manifest_sha256: "2".repeat(64),
            schedule_sha256: "5".repeat(64),
            corpus_manifest_sha256: "6".repeat(64),
            model_verification_sha256: "7".repeat(64),
            worker_sha256: "8".repeat(64),
            executable_sha256: "9".repeat(64),
            source_commit: "3".repeat(40),
            model_key: "12b".into(),
            context_tokens: 4_096,
            generated_tokens: GENERATED_TOKENS,
            arm: arm.into(),
            requested_wired_limit_bytes: cap,
            recommended_working_set_bytes: 12_713_115_648,
            started_unix_ns: started,
            finished_unix_ns: started + 9,
            output_token_sha256: Some("4".repeat(64)),
            trials,
            os_trials,
            validation_errors: Vec::new(),
            failure_recorded: false,
            uncontrolled_oom: false,
            trace_complete: true,
            valid: true,
        }
    }

    fn complete_budget_cells() -> Vec<Cell> {
        const GIB: u64 = 1_073_741_824;
        let mut cells = (4_u64..=11)
            .map(|gib| {
                synthetic_cell(
                    &format!("discovery-coarse-{gib}g"),
                    gib * GIB,
                    if gib == 7 { 100.0 } else { 50.0 },
                    gib * 100,
                )
            })
            .collect::<Vec<_>>();
        cells.push(synthetic_cell(
            &format!("discovery-refine-{}", 7 * GIB - GIB / 2),
            7 * GIB - GIB / 2,
            99.0,
            1_200,
        ));
        cells.push(synthetic_cell(
            &format!("discovery-refine-{}", 7 * GIB + GIB / 2),
            7 * GIB + GIB / 2,
            98.0,
            1_300,
        ));
        cells
    }

    #[test]
    fn budget_schedule_requires_every_point_and_retains_controlled_failures() {
        let mut cells = complete_budget_cells();
        let selected = compute_budget_selection(&cells, true).unwrap();
        assert_eq!(selected.selected_c_bytes, 6_979_321_856);

        let removed = cells.pop().unwrap();
        assert!(compute_budget_selection(&cells, true).is_err());
        cells.push(removed);

        let failure = cells
            .iter_mut()
            .find(|cell| cell.requested_wired_limit_bytes == 4 * 1_073_741_824)
            .unwrap();
        failure.valid = false;
        failure.failure_recorded = true;
        failure.trials.clear();
        failure.os_trials.clear();
        failure.output_token_sha256 = None;
        let selected = compute_budget_selection(&cells, true).unwrap();
        assert!(
            selected.output["points"]
                .as_array()
                .unwrap()
                .iter()
                .any(|point| {
                    point["requested_wired_limit_bytes"] == 4_u64 * 1_073_741_824
                        && point["outcome"] == "failure"
                        && point["eligible"] == false
                })
        );
    }

    #[test]
    fn budget_schedule_rejects_cross_arm_output_drift() {
        let mut cells = complete_budget_cells();
        cells.last_mut().unwrap().output_token_sha256 = Some("9".repeat(64));
        assert!(compute_budget_selection(&cells, true).is_err());
    }

    #[test]
    fn core_matrix_rejects_missing_and_duplicate_cells() {
        let recommended = 12_713_115_648;
        let mut cells = ["12b", "e4b"]
            .into_iter()
            .flat_map(|model| {
                [512_u32, 1_024, 4_096, 8_192, 16_384, 32_768]
                    .into_iter()
                    .map(move |context| {
                        let mut cell =
                            synthetic_cell("core-default", recommended, 10.0, u64::from(context));
                        cell.model_key = model.to_owned();
                        cell.context_tokens = context;
                        cell
                    })
            })
            .collect::<Vec<_>>();
        assert!(validate_core_matrix(&cells).is_ok());
        let removed = cells.pop().unwrap();
        assert!(validate_core_matrix(&cells).is_err());
        cells.push(cells[0].clone());
        assert!(validate_core_matrix(&cells).is_err());
        cells.pop();
        cells.push(removed);
        assert!(validate_core_matrix(&cells).is_ok());
    }

    #[test]
    fn acca_validator_rejects_wrong_candidate_and_reordered_blocks() {
        let recommended = 12_713_115_648;
        let candidate = 6_979_321_856;
        let selection = BudgetResult {
            output: json!({}),
            selected_c_bytes: candidate,
            output_token_sha256: "4".repeat(64),
        };
        let mut blocks = [
            synthetic_cell("acca-a1", recommended, 10.0, 100),
            synthetic_cell("acca-c1", candidate, 20.0, 200),
            synthetic_cell("acca-c2", candidate, 20.0, 300),
            synthetic_cell("acca-a2", recommended, 10.0, 400),
        ];
        assert!(validate_acca_blocks(blocks.each_ref(), &selection).is_ok());
        blocks[1].requested_wired_limit_bytes = candidate + 1;
        assert!(validate_acca_blocks(blocks.each_ref(), &selection).is_err());
        blocks[1].requested_wired_limit_bytes = candidate;
        blocks[2].started_unix_ns = blocks[1].finished_unix_ns;
        assert!(validate_acca_blocks(blocks.each_ref(), &selection).is_err());
    }

    #[test]
    fn server_journal_rebuilds_raw_sse_and_rejects_assembly_drift() {
        let chunks = vec![
            json!({
                "id": "chatcmpl-synthetic",
                "system_fingerprint": "mlx-lm-0.31.3",
                "object": "chat.completion.chunk",
                "model": "default_model",
                "created": 1,
                "choices": [{"index": 0, "delta": {"content": "hello", "role": "assistant"}, "finish_reason": null}],
            }),
            json!({
                "id": "chatcmpl-synthetic",
                "system_fingerprint": "mlx-lm-0.31.3",
                "object": "chat.completion.chunk",
                "model": "default_model",
                "created": 1,
                "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": "stop"}],
            }),
            json!({
                "id": "chatcmpl-synthetic",
                "system_fingerprint": "mlx-lm-0.31.3",
                "object": "chat.completion",
                "model": "default_model",
                "created": 1,
                "choices": [],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            }),
        ];
        let lines = chunks
            .iter()
            .map(|chunk| format!("data: {}", serde_json::to_string(chunk).unwrap()))
            .chain(std::iter::once("data: [DONE]".to_owned()))
            .collect::<Vec<_>>();
        let mut journal = vec![
            json!({"kind": "request", "unix_ns": 1, "body": {"model": "synthetic"}}),
            json!({
                "kind": "response_start",
                "unix_ns": 2,
                "status": 200,
                "content_type": "text/event-stream",
                "cache_control": null,
            }),
        ];
        journal.extend(lines.iter().enumerate().map(|(index, line)| {
            json!({
                "kind": "sse_line",
                "emission_offset_ns": (index + 1) * 10,
                "line": line,
            })
        }));
        journal.push(json!({"kind": "response_end", "unix_ns": 3, "saw_done": true}));
        let path = env::temp_dir().join(format!(
            "hyperion-m1-journal-{}-{}.jsonl",
            std::process::id(),
            unix_time_ns().unwrap()
        ));
        let bytes = journal
            .iter()
            .map(|value| serde_json::to_string(value).unwrap() + "\n")
            .collect::<String>();
        fs::write(&path, bytes).unwrap();
        let mut expected = json!({
            "request": {"model": "synthetic"},
            "status": 200,
            "headers": {"content-type": "text/event-stream", "cache-control": null},
            "raw_lines": lines,
            "chunks": chunks,
            "emission_offsets_ns": [10, 20, 30],
            "http_inter_emission_ns": [10, 10],
            "assembled": {
                "content": "hello world",
                "reasoning_content": "",
                "tool_calls": [],
                "finish_reasons": ["stop"],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        });
        assert!(verify_server_journal(&path, &expected).is_ok());
        expected["assembled"]["content"] = Value::String("tampered".into());
        assert!(verify_server_journal(&path, &expected).is_err());
        assert!(assemble_server_chunks(&[json!({"usage": {}})]).is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn server_exit_rejects_forced_sigkill() {
        assert!(valid_server_exit(&json!({
            "returncode": -15,
            "expected_termination": true,
            "forced_kill": false,
        })));
        assert!(!valid_server_exit(&json!({
            "returncode": -9,
            "expected_termination": true,
            "forced_kill": true,
        })));
    }

    #[test]
    fn server_canonical_hash_matches_python_sorted_json() {
        let first = json!({
            "request": {"messages": [{"role": "user", "content": "synthetic"}]},
            "assembled": {
                "content": "",
                "reasoning_content": "",
                "tool_calls": [{
                    "id": "random-server-id",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "get_points",
                        "arguments": "{\"filter\":\"site:HQ AND equip:AHU-01\"}",
                    },
                }],
                "finish_reasons": ["tool_calls"],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        });
        let final_turn = json!({
            "request": {"messages": [
                {"role": "user", "content": "synthetic"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "random-server-id",
                    "index": 0,
                    "type": "function",
                    "function": {
                        "name": "get_points",
                        "arguments": "{\"filter\":\"site:HQ AND equip:AHU-01\"}",
                    },
                }]},
                {
                    "role": "tool",
                    "tool_call_id": "random-server-id",
                    "name": "get_points",
                    "content": "[{\"id\":\"hq.ahu01.sat\",\"kind\":\"analogInput\",\"label\":\"AHU-01 Supply Air Temperature\",\"tags\":[\"site:HQ\",\"equip:AHU-01\",\"measurement:supply-air-temperature\"],\"unit\":\"degF\"}]",
                },
            ]},
            "assembled": {
                "content": "ok °F",
                "reasoning_content": "",
                "tool_calls": [],
                "finish_reasons": ["stop"],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21},
            },
        });
        let canonical = canonical_server_result(&first, &final_turn).unwrap();
        assert_eq!(
            sha256_bytes(&serde_json::to_vec(&canonical).unwrap()),
            "994b7c9dd04f8f1f69e1ccb42ffe36df6628f16e1238b219f1811898453d88f0"
        );
    }
}
