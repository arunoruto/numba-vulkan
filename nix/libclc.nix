# libclc's Vulkan target (clspv), built from the LLVM sources that nixpkgs
# has for llvmPackages_22. nixpkgs-unstable dropped its own libclc package
# in August 2026 (Mesa moved to a fork without this target), so this is the
# way to get the file once NixOS 26.05 is gone. Only `clspv--.bc` is built.
{
  lib,
  stdenv,
  runCommand,
  buildEnv,
  cmake,
  ninja,
  python3,
  llvmPackages_22,
}:
let
  llvmPackages = llvmPackages_22;
  # libclc's CMake looks for clang, llvm-as, llvm-link and opt in one place.
  tools = buildEnv {
    name = "libclc-tools";
    paths = [
      llvmPackages.clang-unwrapped
      llvmPackages.llvm
    ];
    pathsToLink = [ "/bin" ];
  };
  monorepoSrc = llvmPackages.llvm.monorepoSrc;
in
stdenv.mkDerivation (finalAttrs: {
  pname = "libclc-clspv";
  version = llvmPackages.llvm.version;

  src = runCommand "libclc-src-${finalAttrs.version}" { } ''
    mkdir -p "$out"
    cp -r ${monorepoSrc}/cmake "$out"
    cp -r ${monorepoSrc}/libclc "$out"
  '';
  sourceRoot = "${finalAttrs.src.name}/libclc";

  nativeBuildInputs = [
    cmake
    ninja
    python3
  ];
  buildInputs = [ llvmPackages.llvm ];
  strictDeps = true;

  cmakeFlags = [
    "-DLIBCLC_TARGETS_TO_BUILD=clspv--"
    "-DLIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR=${tools}/bin"
  ];

  meta = {
    description = "libclc's clspv target, the OpenCL math library for Vulkan";
    homepage = "https://libclc.llvm.org/";
    license = with lib.licenses; [
      asl20
      llvm-exception
    ];
    platforms = lib.platforms.all;
  };
})
