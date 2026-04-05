#!/usr/bin/env bash
# ============================================
# MIND EASE — One-Shot Setup Script
# ============================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║    MIND EASE Setup v1.0          ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# 1. Copy env if not exists
if [ ! -f "$ROOT/.env" ]; then
    cp "$ROOT/.env.example" "$ROOT/.env"
    echo "[1/5] Created .env from .env.example"
else
    echo "[1/5] .env already exists"
fi

# 2. Python backend
echo "[2/5] Installing Python backend dependencies..."
cd "$ROOT/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
deactivate
echo "  Backend venv ready"

# 3. Node frontend
echo "[3/5] Installing frontend dependencies..."
cd "$ROOT/frontend"
npm install --silent
echo "  Frontend node_modules ready"

# 4. Build C++ engine
echo "[4/5] Building C++ inference engine..."
bash "$ROOT/scripts/build_engine.sh" || {
    echo "  Warning: Engine build failed. System will run in mock mode."
    echo "  Check scripts/build_engine.sh for details."
}

# 5. Download models
echo "[5/5] Downloading model weights..."
bash "$ROOT/scripts/download_models.sh"

echo ""
echo "Setup complete! To start MIND EASE:"
echo ""
echo "  Terminal 1 (backend):"
echo "    cd backend && source .venv/bin/activate && python main.py"
echo ""
echo "  Terminal 2 (frontend):"
echo "    cd frontend && npm run dev"
echo ""
echo "  Then open: http://localhost:3000"
