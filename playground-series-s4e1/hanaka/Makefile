CURRENT_DIR := $(shell pwd)
PARENT_DIR := $(shell dirname $(CURRENT_DIR))
BASENAME := $(shell basename $(PARENT_DIR))
TIMES_OF := 1

run:
	@echo Watch file changes and run test
	watchmedo shell-command \
    --patterns="*.py;*.txt" \
    --recursive \
		--command='make test' \
    .

watch:
	@$(MAKE) run file=$(file)

submit:
	@echo $(times) Submit to kaggle
	@bash ~/ML_python/ml_utils/count.sh
	@kaggle competitions submit -c $(BASENAME) -f submission.csv -m "submission"

print_parent_dir:
	@echo $(BASENAME)

.PHONY: run watch submit
