# LICOM Makefile - Main entry for LICOM ocean model (JAX-native version)

# Cache settings
CACHE_DIR := .cache
PYTHONPYCACHEPREFIX := $(PWD)/$(CACHE_DIR)
export JAX_ENABLE_X64=1

# Targets
.PHONY: all run field clean help status install version

all: run

# Native JAX run (Single process, multi-device via JAX mesh)
run:
	@mkdir -p $(CACHE_DIR) logs
	@bash -c '\
		echo "=========================================="; \
		echo "Start LICOMpy (JAX-native)..."; \
		echo "=========================================="; \
		export PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX); \
		export XLA_PYTHON_CLIENT_ALLOCATOR=platform; \
		export XLA_FLAGS=--xla_force_host_platform_device_count=6; \
		NX=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"grid\"][\"nx\"]))"); \
		FIELD=field/duogrid_C$${NX}.npz; \
		if [ ! -f "$$FIELD" ]; then \
			echo "[run] Field file $$FIELD not found, generating..."; \
			$(MAKE) field || { echo "[run] ERROR: make field failed"; exit 1; }; \
		fi; \
		PYTHONPATH=src python src/licom/main.py 2>logs/error.log | tee logs/console.log; \
		echo "=========================================="; \
		echo "finished"; \
		echo "=========================================="; \
	'

# Generate initial field (duogrid_C{nx}.npz) using MPI (legacy generator)
field:
	@mkdir -p $(CACHE_DIR) logs field
	@bash -c '\
		echo "=========================================="; \
		echo "Generating initial field..."; \
		echo "=========================================="; \
		NX=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"grid\"][\"nx\"]))"); \
		PX=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"gpu_mesh\"][\"px\"]))"); \
		PY=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"gpu_mesh\"][\"py\"]))"); \
		NP=$$((PX * PY * 6)); \
		echo "Resolution: C$${NX}, PX: $${PX}, PY: $${PY}, Total PEs: $${NP}"; \
		cd field && \
		export PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX); \
		export XLA_PYTHON_CLIENT_ALLOCATOR=platform; \
		export JAX_ENABLE_X64=1; \
		PYTHONPATH=../src/initial_field \
		mpirun -n $${NP} python ../src/initial_field/main.py --nx $${NX} --px $${PX} --py $${PY} \
		2>../logs/field_error.log | cat; \
		echo "=========================================="; \
		echo "Field generation finished"; \
		echo "=========================================="; \
	'

# Show system and module status
status:
	@echo "=========================================="
	@echo "LICOM System Status Check (JAX-native)"
	@echo "=========================================="
	@echo "Python version: $$(python --version)"
	@echo "JAX Devices: $$(python -c "import jax; print(jax.devices())")"
	@echo "=========================================="

# Install dependencies
install:
	@pip install -e src/licom/

# Clean cache and temp files
clean:
	@rm -rf $(CACHE_DIR)
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf logs/*
	@echo "Cleanup complete"

# Help info
help:
	@echo "Targets: run, field, status, install, clean, help"

# Version info
version:
	@echo "LICOM: 0.2.0 (JAX-native)"
