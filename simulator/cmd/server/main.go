package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"

	"github.com/AniruthKarthik/mini-metro/simulator/engine"
	"github.com/AniruthKarthik/mini-metro/simulator/server"
)

func main() {
	addr := flag.String("addr", ":6969", "HTTP address to listen on")
	mapName := flag.String("map", "london", "Map to load: london | nyc | tokyo")
	flag.Parse()

	sim := buildSimulator(*mapName)

	srv := server.New(sim)

	mux := http.NewServeMux()
	srv.RegisterRoutes(mux)

	fmt.Printf("🚇 Mini Metro WebSocket server  map=%s  addr=%s\n", *mapName, *addr)
	go srv.Run()

	log.Fatal(http.ListenAndServe(*addr, withCORS(mux)))
}

// buildSimulator creates a Simulator from a named map config.
func buildSimulator(name string) *engine.Simulator {
	switch name {
	case "nyc", "new_york":
		return engine.NewSimulatorWithMap(engine.NYCMap())
	case "tokyo":
		return engine.NewSimulatorWithMap(engine.TokyoMap())
	default:
		return engine.NewSimulatorWithMap(engine.LondonMap())
	}
}

// withCORS adds permissive CORS headers for local UI development.
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
