.PHONY: dev-build
dev-build:
	docker compose -f docker-compose.yml build

.PHONY: dev-up-d
dev-up-d:
	docker compose -f docker-compose.yml up -d

.PHONY: dev
dev:
	@make dev-build
	@make dev-up-d

.PHONY: down
down:
	docker compose -f docker-compose.yml down