{
  pkgs,
  lib,
  config,
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

  # macOS has no Vulkan driver of its own: MoltenVK implements Vulkan on top
  # of Metal. The loader is pointed at its manifest, and the `vulkan` package,
  # which dlopens libvulkan.dylib by name, finds the loader through dyld.
  env.VK_DRIVER_FILES = lib.mkIf pkgs.stdenv.isDarwin "${pkgs.moltenvk}/share/vulkan/icd.d/MoltenVK_icd.json";
  env.DYLD_FALLBACK_LIBRARY_PATH = lib.mkIf pkgs.stdenv.isDarwin "${pkgs.vulkan-loader}/lib";
  # The SPIR-V backend in llvmlite's macOS arm64 wheel (LLVM 22.1.0)
  # segfaults on common kernels; nixpkgs' llc of the same major version does
  # not (see docs/source/known_issues.md).
  env.NUMBA_VULKAN_LLC = lib.mkIf pkgs.stdenv.isDarwin "${pkgs.llvmPackages_22.llvm}/bin/llc";

  # libclc: LLVM's OpenCL math library as bitcode. numba-vulkan links its
  # functions into kernels for accurate and double-precision math. Built
  # here (nix/libclc.nix) from nixpkgs' LLVM 22 sources, the version inside
  # llvmlite, since nixpkgs no longer packages libclc's Vulkan target.
  env.NUMBA_VULKAN_LIBCLC = "${pkgs.callPackage ./nix/libclc.nix { }}/share/clc/clspv--.bc";

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
    # (/run/opengl-driver on NixOS), including lavapipe as CPU reference;
    # on macOS, from MoltenVK (above).
    libraries = with pkgs; [
      zlib
      stdenv.cc.cc.lib
      vulkan-loader
    ];
  };
}
