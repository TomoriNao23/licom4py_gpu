# LICOM Makefile - Main entry for LICOM ocean model

# Cache settings
CACHE_DIR := .cache
PYTHONPYCACHEPREFIX := $(PWD)/$(CACHE_DIR)
export JAX_ENABLE_X64=1

# Targets
.PHONY: all run field clean debug help status install version

all: run

# Parallel run: npes_x * npes_y * 6 from namelist
run:
	@mkdir -p $(CACHE_DIR) logs
	@bash -c '\
		echo "=========================================="; \
		echo "Start LICOMpy..."; \
		echo "=========================================="; \
		touch input.nml;\
		export PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX); \
        export XLA_PYTHON_CLIENT_ALLOCATOR=platform \
		NX=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"grid\"][\"nx\"]))");\
		FIELD=field/duogrid_C$${NX}.npz; \
		if [ ! -f "$$FIELD" ]; then \
			echo "[run] Field file $$FIELD not found, generating..."; \
			$(MAKE) field || { echo "[run] ERROR: make field failed"; exit 1; }; \
		else \
			echo "[run] Field file $$FIELD found"; \
		fi; \
		PROCS=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"gpu_mesh\"][\"px\"])*int(c[\"gpu_mesh\"][\"py\"])*6)"); \
		PYTHONPATH=src mpirun -n $$PROCS \
		python src/licom/main.py 2>logs/error.log | cat; \
		find . -path ./logs -prune -o -type f \
			\( -name "*.log" -o -name "*.out" \) \
			-maxdepth 2 \
			-exec mv -f {} ./logs/ \; 2>/dev/null || true; \
		rm -f input.nml; \
		echo "=========================================="; \
		echo "finished"; \
		echo "=========================================="; \
	' | tee logs/shell.output

# Generate initial field (duogrid_C{nx}.npz) using MPI
# Resolution read from src/licom/namelist [grid] nx
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
		echo "Resolution: C$${NX}"; \
		cd field && \
		export PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX); \
		export XLA_PYTHON_CLIENT_ALLOCATOR=platform; \
		export JAX_ENABLE_X64=1; \
		PYTHONPATH=../src/initial_field \
		mpirun -n 6 python ../src/initial_field/main.py --nx $${NX} \
		2>../logs/field_error.log | cat; \
		find .. -maxdepth 2 -path ../logs -prune -o \
			-type f \( -name "*.log" -o -name "*.out" \) \
			-exec mv -f {} ../logs/ \; 2>/dev/null || true; \
		echo "=========================================="; \
		echo "Field generation finished"; \
		echo "=========================================="; \
	'
# Debug mode
debug:
	@mkdir -p $(CACHE_DIR) logs
	@bash -c '\
		echo "=========================================="; \
		echo "Start LICOMpy in DEBUG mode..."; \
		echo "=========================================="; \
		export PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX); \
		export XLA_PYTHON_CLIENT_ALLOCATOR=platform \
		PROCS=$$(python -c "import configparser; \
		c=configparser.ConfigParser(); \
		c.read(\"src/licom/namelist\"); \
		print(int(c[\"mpi\"][\"npes_x\"])*int(c[\"mpi\"][\"npes_y\"])*6)"); \
		PYTHONPATH=src \
		mpirun -n $$PROCS \
		python src/licom/main.py --debug \
		2>logs/debug_error.log \
		| tee logs/debug_console.log; \
		find . -path ./logs -prune -o -type f \
			\( -name "*.log" -o -name "*.out" -o -name "*.nml" \) \
			-maxdepth 2 \
			-exec mv -f {} ./logs/ \; 2>/dev/null || true; \
		echo "=========================================="; \
		echo "Debug mode finished"; \
		echo "=========================================="; \
	'

# Show system and module status
status:
	@echo "=========================================="
	@echo "LICOM System Status Check..."
	@echo "=========================================="
	@echo ""
	@echo "Code Statistics:"
	@echo "LICOM Python files:"
	@find src/licom -name "*.py" \
		-not -path "*/__pycache__/*" \
		-not -path "*/.git/*" \
		| wc -l \
		| xargs echo "  Files:"
	@find src/licom -name "*.py" \
		-not -path "*/__pycache__/*" \
		-not -path "*/.git/*" \
		-exec wc -l {} \; \
		| awk '{sum += $$1} END {print "  Total lines (with blanks/comments):", sum}'
	@find src/licom -name "*.py" \
		-not -path "*/__pycache__/*" \
		-not -path "*/.git/*" \
		-exec grep -h "^." {} \; \
		| grep -v "^[[:space:]]*#" \
		| grep -v "^[[:space:]]*\"\"\"[[:space:]]*\"\"\"$$$$" \
		| wc -l \
		| xargs echo "  Code lines (excluding blanks/comments):"
	@echo ""
	@echo "System Information:"
	@echo "Python version:"; python --version
	@echo "Python path: $(shell which python)"
	@echo "mpirun path: $(shell which mpirun)"
	@echo ""
	@echo "Cache directory: $(CACHE_DIR)"
	@echo "Python cache prefix: $(PYTHONPYCACHEPREFIX)"
	@echo ""
	@echo "Installed packages:"
	@pip list \
		| grep -E "(licom|pyfms|numpy|xarray|netcdf4)" \
		|| echo "No relevant packages found"
	@echo ""
	@echo "Module Status:"
	@echo "LICOM module status:"
	@PYTHONPATH=src python -c "import licom; \
		print(f'  Path: {licom.__file__}'); \
		print(f'  Version: {getattr(licom, \"__version__\", \"Not specified\")}')" \
		2>/dev/null \
		|| echo "  LICOM module not found"
	@echo ""
	@echo "pyFMS module status:"
	@PYTHONPATH=src python -c "import pyfms; \
		print(f'  Path: {pyfms.__file__}'); \
		print(f'  Version: {getattr(pyfms, \"__version__\", \"Not specified\")}')" \
		2>/dev/null \
		|| echo "  pyFMS module not found"
	@echo "=========================================="

# Install dependencies
install:
	@echo "=========================================="
	@echo "Installing LICOM dependencies..."
	@echo "=========================================="
	@pip install -e src/licom/
	@pip install -e src/pyFMS/
	@echo "=========================================="
	@echo "Dependencies installation complete"
	@echo "=========================================="

# Clean cache and temp files
clean:
	@echo "=========================================="
	@echo "Cleaning temporary files..."
	@echo "=========================================="
	@rm -rf $(CACHE_DIR)
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@rm -rf logs/*
	@rm -f input.nml
	@echo "Cleanup complete"
	@echo "=========================================="

# Help info
help:
	@echo "=========================================="
	@echo "LICOM Makefile Usage Guide"
	@echo "=========================================="
	@echo "Targets:"
	@echo "  run      - Run LICOM main program (default)"
	@echo "  field    - Generate initial field (duogrid.npz) into field/"
	@echo "  debug    - Run LICOM debug mode (MPI parallel)"
	@echo "  status   - Display system status"
	@echo "  install  - Install dependencies"
	@echo "  clean    - Clean temporary files"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Cache: $(CACHE_DIR)"
	@echo "Python cache: $(PYTHONPYCACHEPREFIX)"
	@echo ""
	@echo "Examples:"
	@echo "  make           # Run LICOM (MPI parallel)"
	@echo "  make debug     # Debug mode (MPI parallel)"
	@echo "  make status    # Status check"
	@echo "=========================================="

# Version info
version:
	@echo "=========================================="
	@echo "LICOM Version Information"
	@echo "=========================================="
	@echo "LICOM: 0.1.0"
	@echo "pyFMS: 2024.2.0"
	@echo "Python: $(shell python --version 2>&1)"
	@echo "Cache directory: $(CACHE_DIR)"
	@echo "=========================================="
