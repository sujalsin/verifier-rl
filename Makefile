PYTHON ?= python3

.PHONY: help reproduce verify demo test check publication publication-preview evidence-check

help:
	@echo "reproduce            Verify public evidence and rebuild results in build/reproduction"
	@echo "verify               Run all public offline checks (install .[test] first)"
	@echo "demo                 Run authored offline cache fixtures"
	@echo "test                 Run all tests (private evidence checks skip when absent)"
	@echo "check                Check repository hygiene, syntax, and whitespace"
	@echo "publication          Alias for reproduce"
	@echo "publication-preview  Build article HTML in build/publication-preview (.[publication])"
	@echo "evidence-check       Require the private evidence bundle and verify it"
	@echo "Override Python with: make test PYTHON=.venv/bin/python"
	@echo "No target launches a cloud experiment or downloads model weights."

reproduce:
	$(PYTHON) scripts/reproduce.py

verify: check demo test reproduce

demo:
	$(PYTHON) -m verifier_rl demo

test:
	$(PYTHON) -m unittest discover -s tests -t .

check:
	$(PYTHON) scripts/check_repository.py
	git diff --check

publication: reproduce

publication-preview:
	mkdir -p build/publication-preview
	cp reports/booking-blog/program_scores.csv reports/booking-blog/source_manifest.json build/publication-preview/
	$(PYTHON) scripts/booking_publication.py --output build/publication-preview --figures --preview

evidence-check:
	$(PYTHON) scripts/check_repository.py --require-evidence
	$(PYTHON) -m unittest tests.test_booking_complete_report tests.test_booking_replication_analysis tests.test_booking_behavior_analysis -v
