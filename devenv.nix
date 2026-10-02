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

  # sdist and wheel with libclc bundled; see the Makefile, which works without Nix.
  scripts.build-dist.exec = ''
    cd "$DEVENV_ROOT" && make dist
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
