.PHONY: test build run clean

test:
	go test -v ./cmd/...

build:
	go build -o bin/simulator ./cmd

run:
	go run ./cmd/main.go

clean:
	rm -rf bin/
