#!/bin/bash
# Cross-compile tract-hello for RISC-V musl (run inside Docker container)
set -e
cd "$(dirname "$0")"
RUSTFLAGS="-C target-feature=+crt-static" \
  cargo build --target riscv64gc-unknown-linux-musl --release
cp target/riscv64gc-unknown-linux-musl/release/tract-hello sh/
