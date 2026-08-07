.PHONY: fnlist

fnlist:
	@find . -type f -name '*.go' \
		-not -path './vendor/*' \
		-not -path './.git/*' | sort | while read -r f; do \
			echo "FILE: $$f"; \
			grep -E '^[[:space:]]*func[[:space:]]+' "$$f" | \
			sed -E 's/^[[:space:]]*func[[:space:]]+/  - /'; \
			echo; \
		done
