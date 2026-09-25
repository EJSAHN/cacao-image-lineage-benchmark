.PHONY: install manifest validate test release
install:
	pip install -e '.[full]'
manifest:
	python scripts/update_manifest.py
validate:
	python scripts/validate_release.py
	cacao-benchmark validate-install
test:
	pytest
release: manifest validate test
	python scripts/build_release.py
