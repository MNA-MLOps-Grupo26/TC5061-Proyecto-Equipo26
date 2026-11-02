#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo26
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests

# DeepChecks 0.19.1 no soporta numpy 2 ni scikit-learn >= 1.6: corre en un
# entorno aislado que uv crea y cachea la primera vez.
DEEPCHECKS = uv run --no-project --python 3.11 --with-requirements validacion_datos/requirements.txt

## Pruebas de los chequeos de DeepChecks con datos sintéticos (no requiere datos)
.PHONY: test-deepchecks
test-deepchecks:
	MIBICI_DEEPCHECKS=1 $(DEEPCHECKS) python -m pytest tests/deepchecks -p no:cacheprovider -W ignore::FutureWarning

## Validar los datos reales con DeepChecks (requiere data/processed)
.PHONY: validar-datos
validar-datos: test-deepchecks
	$(PYTHON_INTERPRETER) -m mibici.validacion
	$(DEEPCHECKS) python -W ignore -m validacion_datos.run_validacion

# Herramientas de seguridad en entornos efímeros de uv: no entran a requirements.txt.
BANDIT = uv run --no-project --python $(PYTHON_VERSION) --with bandit==1.9.4 bandit
PIP_AUDIT = uv run --no-project --python $(PYTHON_VERSION) --with pip-audit==2.10.1 pip-audit
# Riesgo aceptado en el entorno de DeepChecks: corregirlas exige setuptools >= 83,
# que ya no trae pkg_resources. Afectan la construcción de sdists (MANIFEST.in) y
# la instalación legacy de wheels en Windows; ese entorno no hace ninguna.
IGNORAR_SETUPTOOLS = --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w

## Seguridad: código (bandit), dependencias (pip-audit) y secretos versionados
.PHONY: seguridad
seguridad:
	$(BANDIT) -r mibici validacion_datos --severity-level medium
	$(PIP_AUDIT) -r requirements.txt --no-deps --disable-pip --progress-spinner off
	$(DEEPCHECKS) --with pip-audit==2.10.1 pip-audit --progress-spinner off $(IGNORAR_SETUPTOOLS)
	$(PYTHON_INTERPRETER) -m pytest tests/test_seguridad.py

## Pruebas, validación de datos y seguridad en un solo comando
.PHONY: calidad
calidad: test validar-datos seguridad


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mibici/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
