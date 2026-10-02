PYTHON ?= python3

.PHONY: help demo test check publication publication-preview evidence-check

help:
	@echo "demo                 Run authored offline cache fixtures"
	@echo "test                 Run all tests (private evidence checks skip when absent)"
	@echo "check                Check repository hygiene, syntax, and whitespace"
	@echo "publication          Recompute blog arithmetic from the portable score CSV"
	@echo "publication-preview  Build figures and local HTML (publication extra required)"
	@echo "evidence-check       Require the private evidence bundle and verify it"
	@echo "Override Python with: make test PYTHON=.venv/bin/python"
	@echo "No target launches a cloud experiment or downloads model weights."

demo:
	$(PYTHON) -m verifier_rl demo

test:
	$(PYTHON) -m unittest discover -s tests -t .

check:
	$(PYTHON) scripts/check_repository.py
	git diff --check

publication:
	$(PYTHON) scripts/booking_publication.py

publication-preview:
	$(PYTHON) scripts/booking_publication.py --figures --preview

evidence-check:
	$(PYTHON) scripts/check_repository.py --require-evidence
	$(PYTHON) -m unittest tests.test_booking_complete_report tests.test_booking_replication_analysis tests.test_booking_behavior_analysis -v
