.PHONY: back front clean fnlist game build-lib

build-lib:
	@echo "Building Mini Metro C-shared library for Python bindings..."
	@bash ml/build_lib.sh

back:
	@echo "Starting Mini Metro Go backend server on port 6969..."
	cd simulator && go run cmd/server/main.go -addr :6969 -map london

front:
	@echo "Starting Mini Metro Vite frontend server on port 3000..."
	cd ui && npm run dev -- --port 3000 --host

clean:
	@echo "Cleaning processes running on ports 6969 and 3000..."
	@fuser -k 6969/tcp 3000/tcp 2>/dev/null || true
	@lsof -t -i:6969 -i:3000 2>/dev/null | xargs -r kill -9 2>/dev/null || true

fnlist:
	@find . -type f -name '*.go' \
		-not -path './vendor/*' \
		-not -path './.git/*' | sort | while read -r f; do \
			echo "FILE: $$f"; \
			grep -E '^[[:space:]]*func[[:space:]]+' "$$f" | \
			sed -E 's/^[[:space:]]*func[[:space:]]+/  - /'; \
			echo; \
		done

game: clean
	@echo "Starting UI, Backend, and AI. Press Ctrl+C to stop."
	@trap "echo 'Shutting down...'; kill 0" EXIT; \
	(cd ui && npm run dev -- --port 3000 --host 2>&1 | sed -e 's/^/\x1b[36m[UI]\x1b[0m /') & \
	(cd simulator && go run cmd/server/main.go -addr :6969 -map london 2>&1 | sed -e 's/^/\x1b[32m[BACKEND]\x1b[0m /') & \
	(cd ml && source venv/bin/activate && PYTHONUNBUFFERED=1 python agent.py 2>&1 | sed -e 's/^/\x1b[35m[AI]\x1b[0m /') & \
	sleep 3 && (xdg-open http://localhost:3000 2>/dev/null || python -m webbrowser http://localhost:3000); \
	wait
