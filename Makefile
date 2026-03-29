# LICOM Makefile - JAX-native ocean model
# Author  : Chtholly <mengleshan@mail.iap.ac.cn>
# Updated : 2026-03-19
AUTHOR  := Chtholly <mengleshan@mail.iap.ac.cn>
UPDATED := 2026-03-19

# ============================================================
# Directories & paths
# ============================================================
CACHE_DIR := .cache
SRC_LICOM := src/licom/main.py
SRC_FIELD := src/initial_field/main.py
NAMELIST  := scripts/namelist
SCRIPTS   := scripts

# ============================================================
# Read namelist once (NX PX PY PDEV NP)
# ============================================================
_NL       := $(shell python $(SCRIPTS)/read_namelist.py $(NAMELIST))
_NX       := $(word 1,$(_NL))
_PX       := $(word 2,$(_NL))
_PY       := $(word 3,$(_NL))
_PDEV     := $(word 4,$(_NL))
_NP       := $(word 5,$(_NL))
_FIELD_NP := $(word 6,$(_NL))
FIELD_FILE := field/duogrid_C$(_NX).npz

# ============================================================
# JAX / Python environment
# ============================================================
export JAX_ENABLE_X64            = 1
export PYTHONPYCACHEPREFIX       = $(PWD)/$(CACHE_DIR)
export XLA_PYTHON_CLIENT_ALLOCATOR = platform

# ============================================================
# Targets
# ============================================================
.PHONY: all run simulation field _mkdirs _check_field install clean status help version

all: run

# ── run ─────────────────────────────────────────────────────
run: _mkdirs _check_field
	@echo "=========================================="
	@echo "  Start LICOMpy  [run.default]"
	@echo "=========================================="
	@PYTHONPATH=src python $(SRC_LICOM) \
	    2>logs/error.log | tee logs/console.log
	@echo "=========================================="
	@echo "  finished"
	@echo "=========================================="

# ── simulation ──────────────────────────────────────────────
simulation: _mkdirs _check_field
	@echo "=========================================="
	@echo "  Start LICOMpy  [simulation.cpu]"
	@echo "=========================================="
	@XLA_FLAGS=--xla_force_host_platform_device_count=$(_NP) PYTHONPATH=src python $(SRC_LICOM) \
	    2>logs/error.log | tee logs/console.log
	@echo "=========================================="
	@echo "  finished"
	@echo "=========================================="

_check_field:
	@if [ ! -f "$(FIELD_FILE)" ]; then \
	    echo "[run] $(FIELD_FILE) not found, running make field ..."; \
	    $(MAKE) field || { echo "[run] ERROR: make field failed"; exit 1; }; \
	fi

_mkdirs:
	@mkdir -p $(CACHE_DIR) logs field

# ── field ────────────────────────────────────────────────────
field: _mkdirs
	@echo "=========================================="
	@echo "  Generating field  C$(_NX)  PX=$(_PX) PY=$(_PY)  NP=$(_FIELD_NP)"
	@echo "=========================================="
	@cd field && \
	    mpirun -n $(_FIELD_NP) python ../$(SRC_FIELD) \
	        --nx $(_NX) --px $(_PX) --py $(_PY) \
	        2>../logs/field_error.log | cat
	@echo "=========================================="
	@echo "  Field generation finished"
	@echo "=========================================="

# ── utilities ────────────────────────────────────────────────
install:
	@pip install -e src/licom/

clean:
	@rm -rf $(CACHE_DIR)
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf logs/*
	@echo "Cleanup complete"

status:
	@echo "Author  : $(AUTHOR)"
	@echo "Updated : $(UPDATED)"
	@echo "Python  : $$(python --version)"
	@echo "JAX     : $$(python -c 'import jax; print(jax.__version__)')"
	@echo "Devices : $$(python -c 'import jax; print(jax.devices())')"
	@echo "Grid    : C$(_NX)  PX=$(_PX) PY=$(_PY) PDEV=$(_PDEV) NP=$(_NP)"

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  run            Run LICOMpy on GPU"
	@echo "  simulation     Run LICOMpy on CPU (simulate multiple devices)"
	@echo "  field          Generate initial field file"
	@echo "  install        pip install -e src/licom/"
	@echo "  clean          Remove cache, pyc, logs"
	@echo "  status         Show Python/JAX/device info"
	@echo "  version        Print version"

version:
	@echo "LICOM 0.2.0 (JAX-native)"
	@echo "Author  : $(AUTHOR)"
	@echo "Updated : $(UPDATED)"
