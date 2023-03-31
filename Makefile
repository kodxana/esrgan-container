.PHONY: docker
docker:
	docker compose build && docker compose up || true 

.PHONY: docker_break
docker_break:
	docker compose down --rmi all --volumes --remove-orphans
