REFORMAT_DIRS=detexe/

.PHONY: fmt
fmt:
	$(call colorecho, "\n=> Formating files...")
	black $(REFORMAT_DIRS)
	isort $(REFORMAT_DIRS)
	flake8 --ignore E501,E203,E731,W503 $(REFORMAT_DIRS)