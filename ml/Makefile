.PHONY: setup train clean

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup:
	@echo "Setting up virtual environment in ml/venv..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

train: setup
	@echo "Starting training using train.py..."
	$(PYTHON) train.py 2>&1 | tee train.log

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	rm -f train.log
