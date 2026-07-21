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
    process::{Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
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
const ORACLE_LOCK_SHA256: &str = "5e6e51756f1420e078f09badc0748e010eaaa7a5f1b81adfebbe9ca24a0e8883";
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
    default_relative_path: &'static str,
    primary_env: &'static str,
    fallback_env: Option<&'static str>,
}

const MODEL_12B: ModelSpec = ModelSpec {
    key: "12b",
    label: "gemma-4-12B-QAT-Q4-g64-affine",
    manifest_sha256: "9fa3c7f6c49305f621ed1f96edbb34c6402b6229701041db4e607df70e9b4144",
    default_relative_path: "artifacts/models/gemma4-12b-qat-mlx-g64-b4",
    primary_env: "HYPERION_M1_12B_ORACLE_MODEL",
    fallback_env: Some("HYPERION_M0_ORACLE_MODEL"),
};

const MODEL_E4B: ModelSpec = ModelSpec {
    key: "e4b",
    label: "gemma-4-E4B-QAT-Q4-g64-affine",
    manifest_sha256: "9ba65423d3b2bab1e7c52ea88a1a2b0a33c1f51909b1df66330bf872b7a6c2b0",
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
    max_wired_size_bytes: u64,
    max_resident_size_bytes: u64,
    max_phys_footprint_bytes: u64,
    max_interval_phys_footprint_bytes: u64,
    pageins_first: u64,
    pageins_last: u64,
}

#[derive(Debug)]
struct Cell {
    path: PathBuf,
    sha256: String,
    source_commit: String,
    model_key: String,
    context_tokens: u32,
    arm: String,
    requested_wired_limit_bytes: u64,
    trials: Vec<TrialMetrics>,
    os_trials: Vec<OsTrialMetrics>,
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
        _ => Err(Error::new(m1_usage())),
    }
}

/// M1 CLI usage.
#[must_use]
pub fn m1_usage() -> &'static str {
    concat!(
        "usage:\n",
        "  hyperion-bench m1 run-cell --model 12b|e4b --context TOKENS --arm NAME --wired-limit default|BYTES --output benchmarks/raw/m1/FILE.jsonl\n",
        "  hyperion-bench m1 verify-trace PATH\n",
        "  hyperion-bench m1 summarize --input-dir DIR\n",
        "  hyperion-bench m1 select-budget --input-dir DIR\n",
        "  hyperion-bench m1 check-acca --input-dir DIR"
    )
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
    if ![512, 1_024, 4_096, 8_192, 16_384, 32_768].contains(&context) {
        return Err(Error::new(format!(
            "unsupported M1 context {context}; use 512/1024/4096/8192/16384/32768"
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
    let gating_cardinality = generated_tokens == GENERATED_TOKENS;
    let nightly_cardinality =
        generated_tokens == NIGHTLY_GENERATED_TOKENS && context == 512 && arm == "nightly-512x128";
    if (!gating_cardinality && !nightly_cardinality) || warmups != WARMUPS || trials != TRIALS {
        return Err(Error::new(
            "run-cell requires one warmup/five trials and either 1025 generated IDs or the nightly-512x128 129-ID row",
        ));
    }
    let output = validate_raw_output_path(&required("--output")?)?;
    Ok(RunCellArgs {
        model,
        context,
        output,
        arm,
        wired_limit,
        generated_tokens,
        warmups,
        trials,
    })
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

fn run_cell(arguments: RunCellArgs) -> Result<(), Error> {
    let repo = repo_root()?;
    ensure_clean_worktree(&repo)?;
    let source_commit = command_stdout(&repo, "git", &["rev-parse", "HEAD"])?;

    let corpus_dir = repo.join("benchmarks/m1/corpus");
    let corpus_manifest_path = corpus_dir.join("manifest.json");
    let corpus_manifest_sha256 = sha256_file(&corpus_manifest_path)?;
    let manifest: CorpusManifest = serde_json::from_slice(&fs::read(&corpus_manifest_path)?)?;
    if manifest.schema != CORPUS_SCHEMA {
        return Err(Error::new("unsupported M1 corpus manifest schema"));
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

    let output_path = repo.join(&arguments.output);
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
    let stderr_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&stderr_path)
        .map_err(|error| {
            Error::new(format!(
                "refusing to overwrite worker log {}: {error}",
                stderr_path.display()
            ))
        })?;

    let executable = env::current_exe()?;
    write_json_line(
        &mut output,
        &json!({
            "schema": CONTROLLER_SCHEMA,
            "kind": "controller_start",
            "source_commit": source_commit,
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
            "executable_sha256": sha256_file(&executable)?,
            "argv": env::args().collect::<Vec<_>>(),
            "os_sample_period_ms": SAMPLE_PERIOD.as_millis(),
            "worktree_clean": true,
        }),
    )?;

    let mut command = Command::new(&python);
    command
        .current_dir(&repo)
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
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command.spawn()?;
    let child_pid = child.id();
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::new("worker stdout pipe was unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| Error::new("worker stderr pipe was unavailable"))?;
    let stderr_thread = thread::spawn(move || -> std::io::Result<(File, Vec<u8>)> {
        let mut source = BufReader::new(stderr);
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes)?;
        let mut file = stderr_file;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok((file, bytes))
    });

    let done = Arc::new(AtomicBool::new(false));
    let sample_errors = Arc::new(AtomicUsize::new(0));
    let current_trial = Arc::new(Mutex::new(None::<usize>));
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
            Ok(value) => {
                let kind = value.get("kind").and_then(Value::as_str);
                if kind == Some("trial_end") {
                    sample_boundary(
                        child_pid,
                        sampler_started,
                        SampleBoundary::PostTrial,
                        &sample_errors,
                        &current_trial,
                        &samples,
                    );
                }
                if let Err(error) = observe_worker_value(
                    &value,
                    &arguments,
                    &mut counts,
                    &current_trial,
                    &trial_keys,
                ) {
                    validation_errors.push(error.to_string());
                }
                if kind == Some("trial_start") {
                    sample_boundary(
                        child_pid,
                        sampler_started,
                        SampleBoundary::PreTrial,
                        &sample_errors,
                        &current_trial,
                        &samples,
                    );
                }
            }
            Err(error) => validation_errors.push(format!("non-JSON worker stdout: {error}")),
        }
        writeln!(output, "{line}")?;
    }
    done.store(true, Ordering::Release);
    sampler
        .join()
        .map_err(|_| Error::new("process-memory sampler panicked"))?;
    let status = child.wait()?;
    let (_, stderr_bytes) = stderr_thread
        .join()
        .map_err(|_| Error::new("worker stderr collector panicked"))??;

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
    {
        validation_errors.push(format!(
            "worker cardinality mismatch: start={} end={} warmups={} measured={} failures={} hashes={}",
            counts.worker_start,
            counts.worker_end,
            counts.warmups,
            counts.measured,
            counts.failure,
            counts.measured_hashes.len()
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
            "source_commit": source_commit,
            "worker_exit": exit,
            "worker_success": status.success(),
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
    counts: &mut TraceCounts,
    current_trial: &Mutex<Option<usize>>,
    trial_keys: &Mutex<Vec<TrialKey>>,
) -> Result<(), Error> {
    let schema = string_field(value, "schema")?;
    let kind = string_field(value, "kind")?;
    match (schema, kind) {
        (WORKER_SCHEMA, "worker_start") => {
            counts.worker_start += 1;
            if string_field(value, "model_key")? != arguments.model.key
                || u64_field(value, "input_tokens")? != u64::from(arguments.context)
                || u64_field(value, "generated_tokens")? != u64::from(arguments.generated_tokens)
                || string_field(value, "arm")? != arguments.arm
                || value.get("wired_limit_effective").and_then(Value::as_bool) != Some(true)
            {
                return Err(Error::new(
                    "worker-start identity does not match controller request",
                ));
            }
        }
        (WORKER_SCHEMA, "worker_end") => counts.worker_end += 1,
        (WORKER_SCHEMA, "failure") => counts.failure += 1,
        (WORKER_SCHEMA, "model_loaded") => {}
        (WORKER_SCHEMA, "trial_start") => {
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
            *current = Some(keys.len() - 1);
        }
        (WORKER_SCHEMA, "trial_end") => {
            let key = trial_key(value)?;
            let mut current = current_trial
                .lock()
                .map_err(|_| Error::new("current-trial lock poisoned"))?;
            let index = current.ok_or_else(|| Error::new("trial_end without trial_start"))?;
            let keys = trial_keys
                .lock()
                .map_err(|_| Error::new("trial-key lock poisoned"))?;
            if keys.get(index) != Some(&key) {
                return Err(Error::new("trial_end does not match open trial_start"));
            }
            *current = None;
        }
        (TRIAL_SCHEMA, "trial") => {
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
    current_trial: Arc<Mutex<Option<usize>>>,
    samples: Arc<Mutex<Vec<Sample>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        while !done.load(Ordering::Acquire) {
            match hyperion_ffi::sample_process_memory(pid) {
                Ok(memory) => {
                    let elapsed = started.elapsed().as_nanos();
                    let elapsed_ns = u64::try_from(elapsed).unwrap_or(u64::MAX);
                    let trial = current_trial.lock().ok().and_then(|guard| *guard);
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
    current_trial: &Mutex<Option<usize>>,
    samples: &Mutex<Vec<Sample>>,
) {
    let Ok(memory) = hyperion_ffi::sample_process_memory(pid) else {
        errors.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let Ok(trial) = current_trial.lock().map(|guard| *guard) else {
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
    let source = File::open(path)?;
    let mut counts = TraceCounts::default();
    for (index, line) in BufReader::new(source).lines().enumerate() {
        let line = line?;
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| Error::new(format!("line {}: {error}", index + 1)))?;
        if value.get("schema").and_then(Value::as_str) == Some(TRIAL_SCHEMA)
            && value.get("kind").and_then(Value::as_str) == Some("trial")
        {
            let input = u32::try_from(u64_field(&value, "input_tokens")?)
                .map_err(|_| Error::new("input_tokens exceeds u32"))?;
            let generated = u32::try_from(u64_field(&value, "generated_tokens")?)
                .map_err(|_| Error::new("generated_tokens exceeds u32"))?;
            verify_trial(&value, input, generated)?;
            match string_field(&value, "phase")? {
                "warmup" => counts.warmups += 1,
                "measured" => {
                    counts.measured += 1;
                    counts
                        .measured_hashes
                        .insert(string_field(&value, "output_token_sha256")?.to_owned());
                }
                _ => return Err(Error::new("trace has an invalid trial phase")),
            }
        }
    }
    if counts.warmups != WARMUPS || counts.measured != TRIALS || counts.measured_hashes.len() != 1 {
        return Err(Error::new(format!(
            "trace requires one warmup, five measured trials, and one measured output hash; got {counts:?}"
        )));
    }
    println!("M1_TRACE_VALID {}", path.display());
    Ok(())
}

fn summarize(directory: &Path) -> Result<(), Error> {
    let cells = load_cells(directory)?;
    let source_commits = cells
        .iter()
        .map(|cell| cell.source_commit.as_str())
        .collect::<BTreeSet<_>>();
    let summaries = cells
        .iter()
        .map(|cell| -> Result<Value, Error> {
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
            Ok(json!({
                "file": display_raw_path(&cell.path),
                "sha256": cell.sha256,
                "source_commit": cell.source_commit,
                "model_key": cell.model_key,
                "context_tokens": cell.context_tokens,
                "arm": cell.arm,
                "requested_wired_limit_bytes": cell.requested_wired_limit_bytes,
                "trial_count": cell.trials.len(),
                "low_n": cell.trials.len() < usize::try_from(TRIALS).expect("u32 fits usize"),
                "gate_eligible": cell.arm != "nightly-512x128",
                "valid": cell.valid,
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
            }))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let output = json!({
        "schema": "hyperion.m1-summary.v1",
        "input_directory": display_raw_path(directory),
        "source_commits": source_commits,
        "cell_count": cells.len(),
        "cells": summaries,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn select_budget(directory: &Path) -> Result<(), Error> {
    const GIB: u64 = 1_073_741_824;
    const HALF_GIB: u64 = GIB / 2;
    let mut cells = load_cells(directory)?
        .into_iter()
        .filter(|cell| cell.arm.starts_with("discovery-"))
        .collect::<Vec<_>>();
    if cells.len() < 2 {
        return Err(Error::new(
            "budget selection requires at least two discovery cells",
        ));
    }
    cells.sort_by_key(|cell| cell.requested_wired_limit_bytes);
    let mut seen = BTreeSet::new();
    let mut commit = None::<&str>;
    for cell in &cells {
        if !cell.valid
            || cell.model_key != "12b"
            || cell.context_tokens != 4_096
            || cell.trials.len() != usize::try_from(TRIALS).expect("u32 fits usize")
            || !seen.insert(cell.requested_wired_limit_bytes)
        {
            return Err(Error::new(
                "budget discovery contains an invalid, duplicate, non-12B/4K, or low-N cell",
            ));
        }
        if let Some(expected) = commit {
            if cell.source_commit != expected {
                return Err(Error::new("budget discovery spans multiple source commits"));
            }
        } else {
            commit = Some(&cell.source_commit);
        }
    }

    let distributions = cells
        .iter()
        .map(|cell| distribution(cell.trials.iter().map(|trial| trial.decode_tok_s)))
        .collect::<Result<Vec<_>, _>>()?;
    let best_index = empirical_best_index(
        &cells
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
        .min_by_key(|index| cells[**index].requested_wired_limit_bytes)
        .ok_or_else(|| Error::new("budget plateau unexpectedly excluded the empirical best"))?;
    let unique_speed_optimum = distributions
        .iter()
        .enumerate()
        .all(|(index, values)| index == best_index || best.min > values.max);

    let best_cap = cells[best_index].requested_wired_limit_bytes;
    let existing = cells
        .iter()
        .map(|cell| cell.requested_wired_limit_bytes)
        .collect::<BTreeSet<_>>();
    let refinement_candidates = [
        best_cap.checked_sub(HALF_GIB),
        best_cap.checked_add(HALF_GIB),
    ]
    .into_iter()
    .flatten()
    .filter(|candidate| (4 * GIB..=11 * GIB).contains(candidate) && !existing.contains(candidate))
    .collect::<Vec<_>>();
    let points = cells
        .iter()
        .zip(&distributions)
        .map(|(cell, values)| {
            json!({
                "arm": cell.arm,
                "requested_wired_limit_bytes": cell.requested_wired_limit_bytes,
                "decode_tok_s": values,
                "file": display_raw_path(&cell.path),
                "sha256": cell.sha256,
            })
        })
        .collect::<Vec<_>>();
    let output = json!({
        "schema": "hyperion.m1-budget-selection.v1",
        "source_commit": commit,
        "objective": "highest median decode_tok_s",
        "plateau_rule": "median>=99% best and trial ranges overlap",
        "points": points,
        "empirical_best_bytes": best_cap,
        "empirical_best_decode_tok_s": best,
        "unique_speed_optimum": unique_speed_optimum,
        "selection_label": if unique_speed_optimum { "unique" } else { "plateau_or_uncertain" },
        "plateau_bytes": plateau_indices.iter().map(|index| cells[*index].requested_wired_limit_bytes).collect::<Vec<_>>(),
        "selected_c_bytes": cells[selected_index].requested_wired_limit_bytes,
        "refinement_candidates_bytes": refinement_candidates,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn check_acca(directory: &Path) -> Result<(), Error> {
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
    for cell in ordered {
        if !cell.valid
            || cell.model_key != "12b"
            || cell.context_tokens != 4_096
            || cell.source_commit != first.source_commit
            || cell.trials.len() != usize::try_from(TRIALS).expect("u32 fits usize")
        {
            return Err(Error::new(
                "A-C-C-A block is invalid, low-N, wrong workload, or source-mismatched",
            ));
        }
    }
    let a_cap = blocks["acca-a1"].requested_wired_limit_bytes;
    let c_cap = blocks["acca-c1"].requested_wired_limit_bytes;
    if blocks["acca-a2"].requested_wired_limit_bytes != a_cap
        || blocks["acca-c2"].requested_wired_limit_bytes != c_cap
        || a_cap == c_cap
    {
        return Err(Error::new(
            "A-C-C-A cap identities are inconsistent or identical",
        ));
    }
    let a_values = [blocks["acca-a1"], blocks["acca-a2"]]
        .into_iter()
        .flat_map(|cell| cell.trials.iter().map(|trial| trial.decode_tok_s))
        .collect::<Vec<_>>();
    let c_values = [blocks["acca-c1"], blocks["acca-c2"]]
        .into_iter()
        .flat_map(|cell| cell.trials.iter().map(|trial| trial.decode_tok_s))
        .collect::<Vec<_>>();
    let a_distribution = distribution(a_values)?;
    let c_distribution = distribution(c_values)?;
    let speedup_proven = c_distribution.min > a_distribution.max;
    let output = json!({
        "schema": "hyperion.m1-acca.v1",
        "source_commit": first.source_commit,
        "order": ["A", "C", "C", "A"],
        "trials_per_block": TRIALS,
        "a_requested_wired_limit_bytes": a_cap,
        "c_requested_wired_limit_bytes": c_cap,
        "a_decode_tok_s": a_distribution,
        "c_decode_tok_s": c_distribution,
        "candidate_min_gt_baseline_max": speedup_proven,
        "speedup_claim_allowed": speedup_proven,
        "operational_default": if speedup_proven { "C" } else { "A" },
        "blocks": order.iter().map(|arm| json!({
            "arm": arm,
            "file": display_raw_path(&blocks[arm].path),
            "sha256": blocks[arm].sha256,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
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
    let mut source_commit = None::<String>;
    let mut model_key = None::<String>;
    let mut context_tokens = None::<u32>;
    let mut expected_generated_tokens = None::<u32>;
    let mut arm = None::<String>;
    let mut requested_wired_limit_bytes = None::<u64>;
    let mut trials = Vec::<TrialMetrics>::new();
    let mut os_trials = Vec::<OsTrialMetrics>::new();
    let mut warmups = 0_u32;
    let mut valid = false;
    let mut controller_end_count = 0_u32;
    let mut worker_end_count = 0_u32;
    let mut worker_failed = false;
    let mut output_hashes = BTreeSet::new();
    for (line_index, line) in BufReader::new(source).lines().enumerate() {
        let line = line?;
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            Error::new(format!(
                "{} line {} is invalid JSON: {error}",
                path.display(),
                line_index + 1
            ))
        })?;
        match (
            value.get("schema").and_then(Value::as_str),
            value.get("kind").and_then(Value::as_str),
        ) {
            (Some(CONTROLLER_SCHEMA), Some("controller_start")) => {
                if source_commit.is_some() {
                    return Err(Error::new("cell has duplicate controller_start"));
                }
                source_commit = Some(string_field(&value, "source_commit")?.to_owned());
                model_key = Some(string_field(&value, "model_key")?.to_owned());
                context_tokens = Some(
                    u32::try_from(u64_field(&value, "context_tokens")?)
                        .map_err(|_| Error::new("context_tokens exceeds u32"))?,
                );
                expected_generated_tokens = Some(
                    u32::try_from(u64_field(&value, "generated_tokens")?)
                        .map_err(|_| Error::new("generated_tokens exceeds u32"))?,
                );
                arm = Some(string_field(&value, "arm")?.to_owned());
            }
            (Some(WORKER_SCHEMA), Some("worker_start")) => {
                if requested_wired_limit_bytes.is_some() {
                    return Err(Error::new("cell has duplicate worker_start"));
                }
                if value.get("model_key").and_then(Value::as_str) != model_key.as_deref()
                    || value.get("arm").and_then(Value::as_str) != arm.as_deref()
                    || value.get("input_tokens").and_then(Value::as_u64)
                        != context_tokens.map(u64::from)
                    || value.get("generated_tokens").and_then(Value::as_u64)
                        != expected_generated_tokens.map(u64::from)
                    || value.get("wired_limit_effective").and_then(Value::as_bool) != Some(true)
                {
                    return Err(Error::new(
                        "worker envelope differs from the controller envelope",
                    ));
                }
                requested_wired_limit_bytes =
                    Some(u64_field(&value, "requested_wired_limit_bytes")?);
            }
            (Some(TRIAL_SCHEMA), Some("trial")) => {
                let input = u32::try_from(u64_field(&value, "input_tokens")?)
                    .map_err(|_| Error::new("input_tokens exceeds u32"))?;
                let generated = u32::try_from(u64_field(&value, "generated_tokens")?)
                    .map_err(|_| Error::new("generated_tokens exceeds u32"))?;
                if Some(input) != context_tokens || Some(generated) != expected_generated_tokens {
                    return Err(Error::new(
                        "trial cardinality differs from the controller envelope",
                    ));
                }
                verify_trial(&value, input, generated)?;
                match string_field(&value, "phase")? {
                    "warmup" => warmups += 1,
                    "measured" => {
                        output_hashes
                            .insert(string_field(&value, "output_token_sha256")?.to_owned());
                        trials.push(TrialMetrics {
                            trial_index: u32::try_from(u64_field(&value, "trial_index")?)
                                .map_err(|_| Error::new("trial_index exceeds u32"))?,
                            prefill_tok_s: float_field(&value, "prefill_tok_s")?,
                            decode_tok_s: float_field(&value, "decode_tok_s")?,
                            ttft_ns: u64_field(&value, "ttft_ns")?,
                            itl_p50_ns: u64_field(&value, "itl_p50_ns")?,
                            itl_p95_ns: u64_field(&value, "itl_p95_ns")?,
                            itl_p99_ns: u64_field(&value, "itl_p99_ns")?,
                            mlx_active_end_bytes: nested_u64_field(
                                &value,
                                "mlx_memory_end",
                                "active_bytes",
                            )?,
                            mlx_cache_end_bytes: nested_u64_field(
                                &value,
                                "mlx_memory_end",
                                "cache_bytes",
                            )?,
                            mlx_peak_bytes: u64_field(&value, "mlx_peak_bytes")?,
                        });
                    }
                    phase => return Err(Error::new(format!("invalid trial phase: {phase}"))),
                }
            }
            (Some(CONTROLLER_SCHEMA), Some("controller_end")) => {
                controller_end_count += 1;
                valid = value.get("valid").and_then(Value::as_bool) == Some(true)
                    && value
                        .get("validation_errors")
                        .and_then(Value::as_array)
                        .is_some_and(Vec::is_empty);
            }
            (Some(WORKER_SCHEMA), Some("worker_end")) => worker_end_count += 1,
            (Some(OS_SAMPLE_SCHEMA), Some("os_trial_summary"))
                if value.get("phase").and_then(Value::as_str) == Some("measured") =>
            {
                os_trials.push(OsTrialMetrics {
                    trial_index: u32::try_from(u64_field(&value, "trial_index")?)
                        .map_err(|_| Error::new("OS trial_index exceeds u32"))?,
                    sample_count: u64_field(&value, "sample_count")?,
                    max_wired_size_bytes: u64_field(&value, "max_wired_size_bytes")?,
                    max_resident_size_bytes: u64_field(&value, "max_resident_size_bytes")?,
                    max_phys_footprint_bytes: u64_field(&value, "max_phys_footprint_bytes")?,
                    max_interval_phys_footprint_bytes: u64_field(
                        &value,
                        "max_interval_phys_footprint_bytes",
                    )?,
                    pageins_first: u64_field(&value, "pageins_first")?,
                    pageins_last: u64_field(&value, "pageins_last")?,
                });
            }
            (Some(WORKER_SCHEMA), Some("failure")) => worker_failed = true,
            _ => {}
        }
    }
    trials.sort_by_key(|trial| trial.trial_index);
    os_trials.sort_by_key(|trial| trial.trial_index);
    let standard_cardinality = expected_generated_tokens == Some(GENERATED_TOKENS);
    let nightly_cardinality = expected_generated_tokens == Some(NIGHTLY_GENERATED_TOKENS)
        && context_tokens == Some(512)
        && arm.as_deref() == Some("nightly-512x128");
    if (!standard_cardinality && !nightly_cardinality)
        || !matches!(
            context_tokens,
            Some(512 | 1_024 | 4_096 | 8_192 | 16_384 | 32_768)
        )
        || !matches!(model_key.as_deref(), Some("12b" | "e4b"))
        || warmups != WARMUPS
        || trials.len() != usize::try_from(TRIALS).expect("u32 fits usize")
        || trials
            .iter()
            .enumerate()
            .any(|(index, trial)| trial.trial_index != u32::try_from(index).expect("five fits u32"))
        || output_hashes.len() != 1
        || controller_end_count != 1
        || worker_end_count != 1
        || worker_failed
        || os_trials.len() != trials.len()
        || os_trials.iter().enumerate().any(|(index, trial)| {
            trial.trial_index != u32::try_from(index).expect("five fits u32")
                || trial.sample_count < 3
        })
    {
        return Err(Error::new(format!(
            "cell {} fails M1 trial cardinality or determinism",
            path.display()
        )));
    }
    Ok(Cell {
        path: path.to_owned(),
        sha256: sha256_file(path)?,
        source_commit: source_commit.ok_or_else(|| Error::new("cell lacks source commit"))?,
        model_key: model_key.ok_or_else(|| Error::new("cell lacks model key"))?,
        context_tokens: context_tokens.ok_or_else(|| Error::new("cell lacks context"))?,
        arm: arm.ok_or_else(|| Error::new("cell lacks arm"))?,
        requested_wired_limit_bytes: requested_wired_limit_bytes
            .ok_or_else(|| Error::new("cell lacks effective wired limit"))?,
        trials,
        os_trials,
        valid,
    })
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

fn command_stdout(repo: &Path, program: &str, arguments: &[&str]) -> Result<String, Error> {
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
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
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

fn write_json_line(output: &mut File, value: &Value) -> Result<(), Error> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
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
        ]
        .map(str::to_owned);
        assert_eq!(
            parse_run_cell(&nightly).unwrap().generated_tokens,
            NIGHTLY_GENERATED_TOKENS
        );
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
}
