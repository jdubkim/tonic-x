SHELL=/bin/bash
LINT_PATHS=tonic/ tests/ setup.py

tests:
	chmod u+x ./scripts/run_tests.sh
	./scripts/run_tests.sh

lint:
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

.PHONY: tests lint