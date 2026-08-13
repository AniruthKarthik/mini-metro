.PHONY: back front clean fnlist

back:
	@echo "🚇 Starting Mini Metro Go backend server on port 6969..."
	cd simulator && go run cmd/server/main.go -addr :6969 -map london

front:
	@echo "🌐 Starting Mini Metro Vite frontend server on port 3000..."
	cd ui && npm run dev -- --port 3000 --host

clean:
	@echo "🧹 Cleaning processes running on ports 6969 and 3000..."
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
