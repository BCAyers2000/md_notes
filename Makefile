PYTHON ?= python
TEST_ARGS ?= -q

.DEFAULT_GOAL := help
.PHONY: help book chapter lab notebooks test test-external clean

help:
	@echo "Run these commands from the project directory:"
	@echo "  make book       Compile book/build/book.pdf using the supplied figures"
	@echo "  make chapter CH=04  Compile one chapter into book/build/ch04.pdf"
	@echo "  make lab        Start JupyterLab for the notebooks and practice work"
	@echo "  make test       Run mdlab tests and docstring examples"
	@echo "  make test-external  Run optional compiler and simulation-tool checks"
	@echo "  make notebooks  Regenerate notebooks from scripts/notebooks (overwrites them)"
	@echo "  make clean      Remove book compilation files, keeping the PDF"
	@echo "Use PYTHON=/path/to/python to select an interpreter."

book:
	$(MAKE) -C book

chapter:
	$(MAKE) -C book chapter CH=$(CH) PYTHON="$(PYTHON)"

lab:
	$(PYTHON) -m jupyter lab

# Save personal exercise attempts separately before regenerating these files.
notebooks:
	@for script in scripts/notebooks/build_*.py; do \
		$(PYTHON) "$$script" || exit 1; \
	done

test:
	$(PYTHON) -m pytest -c mdlab/pyproject.toml mdlab/tests mdlab/src/mdlab scripts/tests -m "not external" $(TEST_ARGS)

test-external:
	$(PYTHON) -m pytest -c mdlab/pyproject.toml mdlab/tests -m external $(TEST_ARGS)

clean:
	$(MAKE) -C book clean
