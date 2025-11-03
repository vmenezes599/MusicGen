#!/bin/bash

# TTM Server - Docker Management Script
# Usage: ./manage.sh [start|stop|restart|clean]

DOCKER_COMPOSE_FILE="docker-compose.yml"

case "$1" in
status)
    echo " Service Status:"
    docker compose -f $DOCKER_COMPOSE_FILE ps --format table
    echo ""
    echo "🌐 Network Info:"
    docker network ls | grep TTM_server_network
    ;;

logs | l)
    echo "📜 Viewing TTM Server logs..."
    docker compose -f $DOCKER_COMPOSE_FILE logs -f musicgen_server
    ;;

stop | st)
    echo "Stopping TTM Server services..."
    docker compose -f $DOCKER_COMPOSE_FILE down
    echo "✅ Services stopped successfully!"
    ;;

clean | c)
    echo "🧹 Cleaning up Docker resources..."
    docker compose -f $DOCKER_COMPOSE_FILE down -v
    docker system prune -f
    echo "✅ Cleanup completed!"
    ;;

build | b)
    echo "Building TTM Server..."
    docker compose -f $DOCKER_COMPOSE_FILE stop musicgen_server
    docker compose -f $DOCKER_COMPOSE_FILE rm -f musicgen_server
    docker compose -f $DOCKER_COMPOSE_FILE build musicgen_server --no-cache
    docker compose -f $DOCKER_COMPOSE_FILE up -d musicgen_server
    echo "✅ TTM Server build completed!"
    ;;

start | s)
    echo "🚀 Starting TTM Server..."
    docker compose -f $DOCKER_COMPOSE_FILE up -d musicgen_server
    echo "✅ TTM Server started successfully!"
    ;;

*)
    echo "TTM Server - Docker Management"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "📋 BASIC COMMANDS (PRODUCTION):"
    echo "  start|s                         - Start"
    echo "  stop|st                         - Stop"
    echo "  status                          - Status"
    echo ""
    echo "🔨 BUILD:"
    echo "  build|b                         - Build"
    echo "🧹 CLEAN:"
    echo "  clean|c                         - Cleanup resources"
    ;;
esac
