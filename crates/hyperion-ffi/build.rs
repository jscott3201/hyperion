use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("macos"),
        "Hyperion supports macOS only"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("aarch64"),
        "Hyperion supports Apple Silicon only"
    );

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"));
    let native_dir = manifest_dir.join("../../native/hyperion_mlx");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set"));
    let build_dir = out_dir.join("hyperion_mlx-build");
    let mlx_root = find_mlx_root();

    for path in [
        native_dir.join("CMakeLists.txt"),
        native_dir.join("include/hyperion_mlx.h"),
        native_dir.join("metal/canary.metal"),
        native_dir.join("src/platform_policy.cc"),
        native_dir.join("src/platform_policy.h"),
        native_dir.join("src/runtime.mm"),
        native_dir.join("src/abi_error.cc"),
        native_dir.join("src/abi_error.h"),
        native_dir.join("src/geometry.cc"),
        native_dir.join("src/geometry.h"),
        native_dir.join("src/dispatch.cc"),
        native_dir.join("src/dispatch.h"),
        native_dir.join("src/model.cc"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for name in ["MLX_ROOT", "CMAKE_PREFIX_PATH", "MACOSX_DEPLOYMENT_TARGET"] {
        println!("cargo:rerun-if-env-changed={name}");
    }

    fs::create_dir_all(&build_dir).expect("native build directory can be created");
    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&native_dir)
        .arg("-B")
        .arg(&build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_TESTING=OFF")
        .arg("-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2")
        .arg(format!("-DCMAKE_PREFIX_PATH={}", mlx_root.display()));
    run(&mut configure, "configure native hyperion_mlx");

    let mut build = Command::new("cmake");
    build
        .arg("--build")
        .arg(&build_dir)
        .arg("--config")
        .arg("Release")
        .arg("--parallel");
    run(&mut build, "build native hyperion_mlx and metallib");

    copy_metallib_sidecar(&build_dir, &out_dir);

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=hyperion_mlx");
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=objc");
    for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
        println!("cargo:rustc-link-lib=framework={framework}");
    }
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        mlx_root.join("lib").display()
    );
}

fn find_mlx_root() -> PathBuf {
    let mut candidates: Vec<PathBuf> = env::var_os("MLX_ROOT")
        .into_iter()
        .map(PathBuf::from)
        .collect();
    if let Some(paths) = env::var_os("CMAKE_PREFIX_PATH") {
        candidates.extend(env::split_paths(&paths));
    }
    candidates.push(PathBuf::from("/opt/homebrew/opt/mlx"));

    candidates
        .into_iter()
        .find(|root| {
            root.join("lib/libmlx.dylib").is_file()
                && root.join("share/cmake/MLX/MLXConfig.cmake").is_file()
        })
        .unwrap_or_else(|| {
            panic!(
                "MLX 0.32.0 CMake package was not found; set MLX_ROOT to its installation prefix"
            )
        })
}

fn copy_metallib_sidecar(build_dir: &Path, out_dir: &Path) {
    let source = build_dir.join("hyperion_canary.metallib");
    let profile_dir = out_dir
        .ancestors()
        .find(|path| path.file_name().is_some_and(|name| name == "build"))
        .and_then(Path::parent)
        .expect("Cargo OUT_DIR should be nested below a profile build directory");
    let destination = profile_dir.join("hyperion_canary.metallib");
    fs::copy(&source, &destination).unwrap_or_else(|error| {
        panic!(
            "failed to copy {} to {}: {error}",
            source.display(),
            destination.display()
        )
    });
}

fn run(command: &mut Command, action: &str) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to {action}: {error}"));
    assert!(status.success(), "{action} failed with status {status}");
}
