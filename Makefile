# Zig cross-platform builds. See docs/guides/packaging.md for artifacts and GPU acceptance.
.DEFAULT_GOAL := help
PYTHON ?= python3
CARGO ?= cargo
TARGET ?=
CARGO_ZIGBUILD_VERSION := 0.23.4
DIST_DIR ?= artifacts/packages
PACKAGE_FILE ?=
MODEL ?=
GOLDEN ?=
ARCH ?= sm_89
KERNELS_OUT ?= target/kernels/aot

.PHONY: help build local-build test test-package package verify-package accept clean-artifacts
help:
	@echo 'Tests: make test  # the workspace test entries, nothing else'
	@echo 'Zig target pipelines (Linux/CUDA or macOS/Metal):'
	@echo '  make build | test-package | package TARGET=<rust-target>[.<glibc>]'
	@echo '  make local-build  # native release CLI with the platform backend'
	@echo '  make package-linux-cuda | package-macos-metal TARGET=<target>'
	@echo '  make setup-build  # install cargo-zigbuild; requires Zig'
	@echo '  make verify-package PACKAGE_FILE=/path/to/archive.tar.gz'
	@echo '  make accept PACKAGE_FILE=... MODEL=... GOLDEN=...'
	@echo '  DIST_DIR=artifacts/packages overrides the artifact directory.'
	@echo 'CUDA artifacts: make check-cuda-artifacts | list-cuda-kernels | build-kernels-aot ARCH=sm_89'
	@echo 'Iteration: check-iteration (performance + 64 tasks, <=10min); release: check-release'
	@echo 'Quality gates: check, check-rust, check-tools, check-security,'
	@echo '  test-cuda, test-cuda-hosted (CUDA compile/lint/tests, no GPU), test-kernels-aot, test-metal, bench'
	@echo '  (aliases: check-cuda, check-metal, check-cpu),'
	@echo '  check-attention, check-msrv, check-linux, check-linux-numa, check-package'

# `test` is the repository's own test entries. The packaging pipeline's host suite is
# `test-package`, so the two cannot be confused by name.
test:
	./tools/check/gate.sh test

build test-package package:
	$(PYTHON) tools/package/package.py $(patsubst test-package,test,$@) --platform auto --target "$(TARGET)" --out "$(DIST_DIR)"

# Cargo features are compile-time and cannot be selected by the CLI at runtime. The native entry
# point resolves the same plan and preflight as packaging, so a native build and a package of the
# same host cannot disagree about backend or features.
local-build:
	$(PYTHON) tools/package/package.py native

# Recursive Make expansion is deliberately avoided: each package action sequences
# target checks -> Zig release -> archive -> integrity/native smoke -> publication.
define platform_targets
.PHONY: build-$(1) test-$(1) package-$(1)
build-$(1) test-$(1) package-$(1):
	$$(PYTHON) tools/package/package.py $$(word 1,$$(subst -, ,$$@)) --platform $(1) --target "$$(TARGET)" --out "$$(DIST_DIR)"
endef
$(eval $(call platform_targets,linux-cuda))
$(eval $(call platform_targets,macos-metal))

verify-package:
	$(PYTHON) tools/package/package.py verify --archive "$(PACKAGE_FILE)"

accept:
	$(PYTHON) tools/package/package.py accept --archive "$(PACKAGE_FILE)" --model "$(MODEL)" --golden "$(GOLDEN)"

# Device and tool entries use the documented verbs; the check-* names remain as aliases while the
# rest of the tree migrates.
.PHONY: check test-cuda test-cuda-hosted test-kernels-aot test-metal bench check-rust check-tools check-security check-metal check-cuda check-attention check-msrv check-cpu check-linux check-linux-numa check-package
check:
	./tools/check/gate.sh all

test-cuda:
	./tools/check/gate.sh cuda

# Header-only CUDA half: compiles, lints and tests the cuda-gated modules without a device.
test-cuda-hosted:
	./tools/check/gate.sh cuda-hosted

# AOT kernel source: no CUDA toolkit or headers required.
test-kernels-aot:
	./tools/check/gate.sh cuda-aot

test-metal:
	./tools/check/gate.sh metal

bench:
	./tools/check/gate.sh cpu

# Aliases for the migration: same gates, older names.
check-cuda: test-cuda
check-metal: test-metal
check-cpu: bench

check-rust check-tools check-security check-attention check-msrv check-linux check-linux-numa:
	./tools/check/gate.sh $(patsubst check-%,%,$@)

check-package:
	$(PYTHON) -m unittest discover -s tools/package -p 'test_*.py'

clean-artifacts:
	$(PYTHON) tools/check/clean-artifacts.py

.PHONY: setup-build
setup-build:
	cargo install --locked cargo-zigbuild --version $(CARGO_ZIGBUILD_VERSION)
	"$${CARGO_ZIGBUILD_ZIG_PATH:-zig}" version

# Artifact integrity is portable and does not require a CUDA Toolkit or GPU.
.PHONY: check-cuda-artifacts list-cuda-kernels build-kernels-aot
check-cuda-artifacts:
	$(PYTHON) tools/cuda/kernels.py verify

list-cuda-kernels:
	$(PYTHON) tools/cuda/kernels.py list

# Build one AOT artifact set (cubins + manifest). Produces a package sidecar, not a source file.
build-kernels-aot:
	$(PYTHON) tools/cuda/build_aot.py --arch $(ARCH) --out $(KERNELS_OUT)

# Iteration snapshots use prebuilt binaries and never start formal release measurement.
ITERATION_ARGS ?=
RELEASE_ARGS ?=
.PHONY: check-iteration check-release
check-iteration:
	@test -n "$(ITERATION_ARGS)" || { echo 'Set ITERATION_ARGS; see docs/guides/quality-gates.md'; exit 2; }
	bash tools/bench/safe-run.sh --memory-gib 24 --time-limit-seconds 600 $(PYTHON) tools/bench/iteration-gate.py $(ITERATION_ARGS)

# Full frozen process-pair evidence and the complete quality suite belong here.
check-release:
	@test -n "$(RELEASE_ARGS)" || { echo 'Set RELEASE_ARGS; see docs/guides/quality-gates.md'; exit 2; }
	bash tools/bench/safe-run.sh --memory-gib 24 $(PYTHON) tools/bench/release-gate.py $(RELEASE_ARGS)
