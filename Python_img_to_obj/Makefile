.PHONY: test smoke lint clean install real

install:
	python -m pip install -e .

real:
	python -m pip install -e ".[real]"

test:
	pytest -q

smoke:
	python -m pipeline.run single --image tests/fixtures/person_stub.png --out runs/smoke --backend dummy --quick
	pytest -q tests/test_cli_smoke.py

lint:
	python -m compileall src tests

clean:
	rm -rf runs/smoke .pytest_cache
