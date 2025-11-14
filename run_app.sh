#!/bin/bash
################################################################################
# Medical NLP Lean Package - Streamlit App Launcher
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="py311_bionlp"
APP_FILE="$SCRIPT_DIR/app/medical_nlp_app.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default port
PORT="${1:-8501}"

# Function to print colored header
print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to display app already running message
show_app_running() {
    echo ""
    print_header "🚀 Medical NLP Streamlit App Status"
    echo ""
    echo -e "${GREEN}✓${NC} ${BOLD}App is already running!${NC}"
    echo ""
    echo -e "${CYAN}📍 Port:${NC}     ${BOLD}$PORT${NC}"
    echo -e "${CYAN}🌐 URL:${NC}      ${BOLD}${BLUE}http://localhost:$PORT${NC}"
    echo ""
    echo -e "${YELLOW}ℹ️  The application is accessible at the URL above.${NC}"
    echo -e "${YELLOW}   Simply open the link in your web browser.${NC}"
    echo ""
    echo -e "${MAGENTA}💡 Tips:${NC}"
    echo -e "   • Press ${BOLD}Ctrl+C${NC} in the terminal running the app to stop it"
    echo -e "   • Use a different port: ${BOLD}./run_app.sh 8502${NC}"
    echo -e "   • Check running process: ${BOLD}lsof -i :$PORT${NC}"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to launch app
launch_app() {
    echo ""
    print_header "🚀 Launching Medical NLP Streamlit App"
    echo ""
    echo -e "${CYAN}📂 App File:${NC}  $APP_FILE"
    echo -e "${CYAN}📍 Port:${NC}      ${BOLD}$PORT${NC}"
    echo -e "${CYAN}🌐 URL:${NC}       ${BOLD}${BLUE}http://localhost:$PORT${NC}"
    echo ""
    echo -e "${GREEN}✓${NC} Starting application..."
    echo -e "${YELLOW}  (Press Ctrl+C to stop)${NC}"
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Launch streamlit
    conda run -n "$CONDA_ENV" streamlit run "$APP_FILE" --server.port="$PORT" 2>&1 | while IFS= read -r line; do
        # Check if the line indicates port already in use
        if echo "$line" | grep -q "Port.*is already in use"; then
            # This should not happen since we checked, but handle it anyway
            echo ""
            show_app_running
            return 1
        fi
        # Print other output normally
        echo "$line"
    done
}

# Main execution
main() {
    # Check if app file exists
    if [[ ! -f "$APP_FILE" ]]; then
        echo -e "${RED}✗ Error: App file not found: $APP_FILE${NC}"
        exit 1
    fi

    # Check if port is already in use
    if check_port "$PORT"; then
        show_app_running

        # Open browser automatically
        echo -e "${CYAN}🔗 Opening browser...${NC}"
        sleep 1

        # Try to open in default browser (cross-platform)
        if command -v open >/dev/null 2>&1; then
            # macOS
            open "http://localhost:$PORT" 2>/dev/null &
        elif command -v xdg-open >/dev/null 2>&1; then
            # Linux
            xdg-open "http://localhost:$PORT" 2>/dev/null &
        elif command -v start >/dev/null 2>&1; then
            # Windows (Git Bash)
            start "http://localhost:$PORT" 2>/dev/null &
        fi

        echo -e "${GREEN}✓${NC} Browser opened (if available)"
        echo ""
        exit 0
    fi

    # Port is free, launch the app
    launch_app
}

# Run main function
main
