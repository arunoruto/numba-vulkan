{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  packages = [
    pkgs.git
    # spirv-val / spirv-dis / spirv-as: validate and inspect generated shaders
    pkgs.spirv-tools
    # vulkaninfo
    pkgs.vulkan-tools
  ];

  # libclc: LLVM's OpenCL math library as bitcode. numba-vulkan links its
  # functions into kernels for accurate and double-precision math. The
  # version must not be newer than the LLVM inside llvmlite (22).
  env.NUMBA_VULKAN_LIBCLC =
    let
      libclcPkgs = inputs.nixpkgs-libclc.legacyPackages.${pkgs.stdenv.hostPlatform.system};
    in
    "${libclcPkgs.llvmPackages_22.libclc}/share/clc/clspv--.bc";

  # Building distributions: the wheel ships libclc, so that `pip install`
  # needs nothing else. The bitcode is copied from the Nix store into the
  # package directory (ignored by git) before the build.
  scripts.bundle-libclc.exec = ''
    install -Dm644 "$NUMBA_VULKAN_LIBCLC" \
      "$DEVENV_ROOT/src/numba_vulkan/data/clspv--.bc"
  '';
  scripts.build-dist.exec = ''
    bundle-libclc && cd "$DEVENV_ROOT" && uv build "$@"
  '';

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi

    # numba-cuda (benchmark comparison only) needs the host's libcuda.so.
    if [ -e /run/opengl-driver/lib/libcuda.so.1 ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
    fi
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        groups = [
          "test"
          "bench"
          "docs"
        ];
      };
    };

    # PyPI wheels (llvmlite, numpy) need these at runtime; the `vulkan`
    # package dlopens libvulkan.so.1. GPU drivers (ICDs) come from the host
    # (/run/opengl-driver on NixOS), including lavapipe as CPU reference.
    libraries = with pkgs; [
      zlib
      stdenv.cc.cc.lib
      vulkan-loader
    ];
  };
}
