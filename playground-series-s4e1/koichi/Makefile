CURRENT_DIR := $(shell pwd)
PARENT_DIR := $(shell dirname $(CURRENT_DIR))
BASENAME := $(shell basename $(PARENT_DIR))

run:
	@echo Watch file changes and run test
	watchmedo shell-command \
    --patterns="*.py;*.txt" \
    --recursive \
		--command='make test' \
    .

watch:
	@$(MAKE) run file=$(file)

test:
	@echo Run test $(file)
	python $(file)

download:
	@echo Download data
	@kaggle competitions download -c $(BASENAME)
	@unzip $(BASENAME).zip
	@rm $(BASENAME).zip

submit:
	@echo Submit to kaggle
	@bash ~/ML_python/MLBCU/utils/submit.sh $(BASENAME)

print_parent_dir:
	@echo $(BASENAME)

.PHONY: run watch test submit download print_parent_dir
