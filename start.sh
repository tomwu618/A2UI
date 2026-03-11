#!/bin/bash

# =============================================================================
# A2UI Project Startup Script
# =============================================================================
# This script starts both backend (Agent) and frontend (Shell) servers
# Uses OpenAI-compatible LLM API
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT_RESTAURANT=10002
BACKEND_PORT_CONTACT=10003
FRONTEND_PORT=5173
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR_RESTAURANT="$PROJECT_ROOT/samples/agent/adk/restaurant_finder"
BACKEND_DIR_CONTACT="$PROJECT_ROOT/samples/agent/adk/contact_lookup"
FRONTEND_DIR="$PROJECT_ROOT/samples/client/lit/shell"
AGENT_SDK_DIR="$PROJECT_ROOT/agent_sdks/python/src"

# LLM Configuration (OpenAI-compatible)
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL}"
export LLM_MODEL_NAME="${LLM_MODEL_NAME}"

# PIDs for cleanup
BACKEND_PID_RESTAURANT=""
BACKEND_PID_CONTACT=""
FRONTEND_PID=""

# =============================================================================
# Cleanup function (called on script exit)
# =============================================================================
cleanup() {
    echo -e "\n${YELLOW}Shutting down A2UI servers...${NC}"

    if [ -n "$BACKEND_PID_RESTAURANT" ] && kill -0 "$BACKEND_PID_RESTAURANT" 2>/dev/null; then
        echo -e "  ${RED}→${NC} Stopping Restaurant Agent (PID: $BACKEND_PID_RESTAURANT)"
        kill "$BACKEND_PID_RESTAURANT" 2>/dev/null || true
    fi

    if [ -n "$BACKEND_PID_CONTACT" ] && kill -0 "$BACKEND_PID_CONTACT" 2>/dev/null; then
        echo -e "  ${RED}→${NC} Stopping Contact Agent (PID: $BACKEND_PID_CONTACT)"
        kill "$BACKEND_PID_CONTACT" 2>/dev/null || true
    fi

    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "  ${RED}→${NC} Stopping Frontend Shell (PID: $FRONTEND_PID)"
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    # Also kill any orphaned processes on our ports
    pkill -f "__main__.py --host localhost --port $BACKEND_PORT_RESTAURANT" 2>/dev/null || true
    pkill -f "__main__.py --host localhost --port $BACKEND_PORT_CONTACT" 2>/dev/null || true
    pkill -f "vite.*$FRONTEND_PORT" 2>/dev/null || true

    echo -e "${GREEN}All servers stopped.${NC}"
    exit 0
}

# Set trap for cleanup on SIGINT, SIGTERM, or script exit
trap cleanup SIGINT SIGTERM EXIT

# =============================================================================
# Check prerequisites
# =============================================================================
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  A2UI Project Startup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed${NC}"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: node is not installed${NC}"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Prerequisites checked"

# =============================================================================
# Check if ports are already in use
# =============================================================================
check_port() {
    local port=$1
    local pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Warning: Port $port is already in use (PID: $pid)${NC}"
        echo -e "  Attempting to kill existing process..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        if lsof -ti :$port &>/dev/null; then
            echo -e "${RED}Error: Could not free port $port${NC}"
            return 1
        fi
        echo -e "${GREEN}  ✓${NC} Port $port freed"
    fi
    return 0
}

check_port $BACKEND_PORT_RESTAURANT || exit 1
check_port $BACKEND_PORT_CONTACT || exit 1
check_port $FRONTEND_PORT || exit 1

# =============================================================================
# Start Backend Servers
# =============================================================================
echo ""
echo -e "${BLUE}Starting Backend Agent Servers...${NC}"

# Start Restaurant Finder Agent
echo -e "  ${YELLOW}→${NC} Restaurant Finder (Port: $BACKEND_PORT_RESTAURANT)"
cd "$BACKEND_DIR_RESTAURANT"
export PYTHONPATH="$AGENT_SDK_DIR:$BACKEND_DIR_RESTAURANT"
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export LLM_MODEL_NAME="$LLM_MODEL_NAME"

python3 __main__.py --host localhost --port $BACKEND_PORT_RESTAURANT > /tmp/a2ui_backend_restaurant.log 2>&1 &
BACKEND_PID_RESTAURANT=$!

# Start Contact Lookup Agent
echo -e "  ${YELLOW}→${NC} Contact Lookup (Port: $BACKEND_PORT_CONTACT)"
cd "$BACKEND_DIR_CONTACT"
export PYTHONPATH="$AGENT_SDK_DIR:$BACKEND_DIR_CONTACT"

python3 __main__.py --host localhost --port $BACKEND_PORT_CONTACT > /tmp/a2ui_backend_contact.log 2>&1 &
BACKEND_PID_CONTACT=$!

# Wait for backends to start
echo -e "  ${YELLOW}→${NC} Waiting for backends to be ready..."
for i in {1..30}; do
    restaurant_ready=false
    contact_ready=false

    if curl -s "http://localhost:$BACKEND_PORT_RESTAURANT/" > /dev/null 2>&1; then
        restaurant_ready=true
    fi

    if curl -s "http://localhost:$BACKEND_PORT_CONTACT/" > /dev/null 2>&1; then
        contact_ready=true
    fi

    if $restaurant_ready && $contact_ready; then
        echo -e "  ${GREEN}✓${NC} Both backend servers started"
        break
    fi

    # Check if processes are still running
    if ! kill -0 "$BACKEND_PID_RESTAURANT" 2>/dev/null && ! kill -0 "$BACKEND_PID_CONTACT" 2>/dev/null; then
        echo -e "${RED}Error: Backend servers failed to start${NC}"
        echo -e "${RED}Restaurant logs:${NC}"
        cat /tmp/a2ui_backend_restaurant.log
        echo -e "${RED}Contact logs:${NC}"
        cat /tmp/a2ui_backend_contact.log
        exit 1
    fi
    sleep 0.5
done

if [ $i -eq 30 ]; then
    echo -e "${YELLOW}Warning: Some backend servers may not be fully ready yet${NC}"
fi

echo -e "  ${GREEN}✓${NC} Restaurant Agent started (PID: $BACKEND_PID_RESTAURANT)"
echo -e "  ${GREEN}✓${NC} Contact Agent started (PID: $BACKEND_PID_CONTACT)"

# =============================================================================
# Start Frontend Server
# =============================================================================
echo ""
echo -e "${BLUE}Starting Frontend Shell Server...${NC}"

cd "$FRONTEND_DIR"
npm run dev > /tmp/a2ui_frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start and extract the actual port
echo -e "  ${YELLOW}→${NC} Waiting for frontend to be ready..."
sleep 5

# Try to find the port from the log
ACTUAL_FRONTEND_PORT=$(grep -oE "localhost:[0-9]+" /tmp/a2ui_frontend.log 2>/dev/null | head -1 | cut -d: -f2)
if [ -z "$ACTUAL_FRONTEND_PORT" ]; then
    ACTUAL_FRONTEND_PORT=$FRONTEND_PORT
fi

# Check if frontend is running
for i in {1..20}; do
    if curl -s "http://localhost:$ACTUAL_FRONTEND_PORT/" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Frontend server started (PID: $FRONTEND_PID)"
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${RED}Error: Frontend server failed to start${NC}"
        echo -e "${RED}Logs:${NC}"
        cat /tmp/a2ui_frontend.log
        exit 1
    fi
    sleep 0.5
done

# =============================================================================
# Print startup information
# =============================================================================
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  A2UI Project Started Successfully!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  • LLM Model:    ${YELLOW}$LLM_MODEL_NAME${NC}"
echo -e "  • Base URL:     ${YELLOW}$OPENAI_BASE_URL${NC}"
echo -e "  • API Key:      ${YELLOW}${OPENAI_API_KEY:0:15}...${NC}"
echo ""
echo -e "${BLUE}Servers:${NC}"
echo -e "  • Restaurant Agent: ${GREEN}http://localhost:$BACKEND_PORT_RESTAURANT${NC} (PID: $BACKEND_PID_RESTAURANT)"
echo -e "  • Contact Agent:    ${GREEN}http://localhost:$BACKEND_PORT_CONTACT${NC} (PID: $BACKEND_PID_CONTACT)"
echo -e "  • Frontend Shell:   ${GREEN}http://localhost:$ACTUAL_FRONTEND_PORT${NC} (PID: $FRONTEND_PID)"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
echo -e "  • Restaurant Finder: ${GREEN}http://localhost:$ACTUAL_FRONTEND_PORT/?app=restaurant${NC}"
echo -e "  • Contact Lookup:    ${GREEN}http://localhost:$ACTUAL_FRONTEND_PORT/?app=contacts${NC}"
echo ""
echo -e "${YELLOW}To stop: Press Ctrl+C or run 'pkill -f start.sh'${NC}"
echo ""

# Keep script running and monitor processes
while true; do
    if ! kill -0 "$BACKEND_PID_RESTAURANT" 2>/dev/null && ! kill -0 "$BACKEND_PID_CONTACT" 2>/dev/null && ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "${RED}All servers stopped unexpectedly${NC}"
        exit 1
    fi
    sleep 5
done
