.PHONY: back front clean fnlist game build-lib test ml algo ml-game algo-game _run_game

POLICY_FLAG := .agent_policy_mode

build-lib:
	@echo "Building Mini Metro C-shared library for Python bindings..."
	@bash ml/build_lib.sh

test: build-lib
	@echo "Running Go engine test suite..."
	cd simulator && go test -v ./...
	@echo "Running Python test suites..."
	PYTHONPATH=. ./ml/venv/bin/python -m unittest discover -s ml -p "test_*.py"
	@echo "All tests passed successfully!"

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

ml:
	@echo "model" > $(POLICY_FLAG)

algo:
	@echo "grandmaster" > $(POLICY_FLAG)

ml-game: clean
	@echo "model" > $(POLICY_FLAG)
	@$(MAKE) _run_game

algo-game: clean
	@echo "grandmaster" > $(POLICY_FLAG)
	@$(MAKE) _run_game

game: clean
	@$(MAKE) _run_game

_run_game:
	@MODE=$$(cat $(POLICY_FLAG) 2>/dev/null || echo "grandmaster"); \
	rm -f $(POLICY_FLAG); \
	if [ "$$MODE" = "model" ]; then \
		echo ""; \
		echo "================================================================================"; \
		echo "🚇 STARTING MINI METRO: DEEP REINFORCEMENT LEARNING (RL) AGENT MODE"; \
		echo "🧠 Model Policy: Graph Attention Network Actor-Critic (PPO Checkpoint)"; \
		echo "📁 Loading checkpoint from ml/runs/minimetro_ppo..."; \
		echo "================================================================================"; \
		echo ""; \
	else \
		echo ""; \
		echo "================================================================================"; \
		echo "🚇 STARTING MINI METRO: GRANDMASTER ALGORITHMIC CONTROLLER MODE"; \
		echo "🏆 Strategy: Compact Headway (≤5 st/line), Proactive Hubs, Dynamic Crisis Relief"; \
		echo "🎯 Target Transit Performance: > 300 Passengers"; \
		echo "================================================================================"; \
		echo ""; \
	fi; \
	echo "Starting UI, Backend, and AI ($$MODE). Press Ctrl+C to stop."; \
	trap "echo 'Shutting down...'; kill 0" EXIT; \
	(cd ui && npm run dev -- --port 3000 --host 2>&1 | sed -e 's/^/\x1b[36m[UI]\x1b[0m /') & \
	(cd simulator && go run cmd/server/main.go -addr :6969 -map london 2>&1 | sed -e 's/^/\x1b[32m[BACKEND]\x1b[0m /') & \
	(cd ml && source venv/bin/activate && PYTHONUNBUFFERED=1 python agent.py --policy $$MODE 2>&1 | sed -e 's/^/\x1b[35m[AI]\x1b[0m /') & \
	sleep 3 && (xdg-open http://localhost:3000 2>/dev/null || python -m webbrowser http://localhost:3000); \
	wait
