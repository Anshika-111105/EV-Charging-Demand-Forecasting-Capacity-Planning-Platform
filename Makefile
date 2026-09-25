.PHONY: install ingest validate preprocess features train backtest evaluate forecast test test-unit test-integration lint format docker-build docker-up dashboard api clean

PYTHON := python
PIP := pip

install:
	$(PIP) install -e .[dev]

ingest:
	$(PYTHON) pipelines/ingest_pipeline.py --config configs/development.yaml

preprocess:
	$(PYTHON) pipelines/preprocessing_pipeline.py --config configs/development.yaml

train:
	$(PYTHON) pipelines/training_pipeline.py --config configs/development.yaml

forecast:
	$(PYTHON) pipelines/forecasting_pipeline.py --config configs/development.yaml

monitoring:
	$(PYTHON) pipelines/monitoring_pipeline.py --config configs/development.yaml

test:
	pytest tests/ -v --cov=src/ev_forecasting --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

lint:
	ruff check .

format:
	ruff format .

api:
	uvicorn src.ev_forecasting.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run app/streamlit_app.py --server.port 8501

docker-build:
	docker build -t ev-forecasting:latest .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
