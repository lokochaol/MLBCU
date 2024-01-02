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

submit:
	@echo $(times) Submit to kaggle
	@kaggle competitions submit -c $(project) -f submission.csv -m "$(times) submission"

print_parent_dir:
	@echo $(BASENAME)

.PHONY: run watch submit
