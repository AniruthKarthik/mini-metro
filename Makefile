.PHONY: back front clean fnlist game build-lib test train train-local finetune finetune-berlin gpu-game cpu-game game-gpu game-cpu gpu cpu ml-game

build-lib:
	@echo "Building Mini Metro C-shared library for Python bindings..."
	@bash ml/build_lib.sh

test: build-lib
	@echo "Running Go engine test suite..."
	cd simulator && go test -v ./...
	@echo "All tests passed successfully!"

train: build-lib
	@echo "Starting multi-map PPO training across London, NYC, Tokyo, and Berlin..."
	cd ml && ./venv/bin/python train.py --maps 0 1 2 3 --map-mode stratified

train-local: build-lib
	@echo "Starting fast local CPU multi-map training across all 4 maps..."
	cd ml && ./venv/bin/python train_local.py --maps 0 1 2 3 --map-mode stratified

finetune: build-lib
	@echo "Fine-tuning existing model across all 4 maps (London, NYC, Tokyo, Berlin)..."
	cd ml && ./venv/bin/python train.py --fine-tune --maps 0 1 2 3 --map-mode stratified

finetune-berlin: build-lib
	@echo "Fine-tuning existing model with emphasis on the new Berlin map..."
	cd ml && ./venv/bin/python train.py --fine-tune --maps 0 1 2 3 --map-mode mixed --map-weights 1 1 1 3

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

game: gpu-game
ml-game: gpu-game
game-gpu: gpu-game
game-cpu: cpu-game
gpu: gpu-game
cpu: cpu-game

gpu-game: clean
	@echo ""
	@echo "================================================================================"
	@echo "STARTING MINI METRO: ACTUAL TRAINED MODEL (GPU / 256-DIM PPO)"
	@echo "Model Policy: ml/runs/minimetro_ppo/model_final.pt (hidden_dim=256)"
	@echo "Initial Stations: Randomized Spawning"
	@echo "================================================================================"
	@echo ""
	@echo "Starting UI, Backend, and AI (Actual 256-dim Model). Press Ctrl+C to stop."
	@trap "echo 'Shutting down...'; kill 0" EXIT; \
	(cd ui && npm run dev -- --port 3000 --host 2>&1 | sed -e 's/^/\x1b[36m[UI]\x1b[0m /') & \
	(cd simulator && go run cmd/server/main.go -addr :6969 -map london 2>&1 | sed -e 's/^/\x1b[32m[BACKEND]\x1b[0m /') & \
	(cd ml && source venv/bin/activate && PYTHONUNBUFFERED=1 python agent.py --model runs/minimetro_ppo/model_final.pt 2>&1 | sed -e 's/^/\x1b[35m[AI]\x1b[0m /') & \
	sleep 3 && (xdg-open http://localhost:3000 2>/dev/null || python -m webbrowser http://localhost:3000); \
	wait

cpu-game: clean
	@echo ""
	@echo "================================================================================"
	@echo "STARTING MINI METRO: LOCAL CPU TRAINED MODEL (32-DIM PPO)"
	@echo "Model Policy: ml/runs/minimetro_ppo_local/model_final.pt (hidden_dim=32)"
	@echo "Initial Stations: Randomized Spawning"
	@echo "================================================================================"
	@echo ""
	@echo "Starting UI, Backend, and AI (CPU 32-dim Model). Press Ctrl+C to stop."
	@trap "echo 'Shutting down...'; kill 0" EXIT; \
	(cd ui && npm run dev -- --port 3000 --host 2>&1 | sed -e 's/^/\x1b[36m[UI]\x1b[0m /') & \
	(cd simulator && go run cmd/server/main.go -addr :6969 -map london 2>&1 | sed -e 's/^/\x1b[32m[BACKEND]\x1b[0m /') & \
	(cd ml && source venv/bin/activate && PYTHONUNBUFFERED=1 python agent.py --model runs/minimetro_ppo_local/model_final.pt --device cpu 2>&1 | sed -e 's/^/\x1b[35m[AI]\x1b[0m /') & \
	sleep 3 && (xdg-open http://localhost:3000 2>/dev/null || python -m webbrowser http://localhost:3000); \
	wait
