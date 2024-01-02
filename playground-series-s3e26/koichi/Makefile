watch:
	@echo Watch file changes and run test
	watchmedo shell-command \
    --patterns="*.py;*.txt" \
    --recursive \
		--command='make test' \
    .

run:
	@$(MAKE) watch file=$(file)

test:
	@echo Run test $(file)
	python $(file)

.PHONY: run test watch
