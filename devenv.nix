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
