#!/usr/bin/env bash
set -euo pipefail

echo "========================================="
echo "  Environment Setup Script"
echo "  - uv + Python 3.11 + Claude Code"
echo "========================================="

# ── 1. Install uv ────────────────────────────────────────────────────────────
echo ""
echo "[1/4] Installing uv..."
if command -v uv &>/dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  uv installed: $(uv --version)"
fi

# Ensure uv is on PATH for the rest of the script
export PATH="$HOME/.local/bin:$PATH"

# ── 2. Install Python 3.11 via uv ──────────────────────────────────────────
echo ""
echo "[2/4] Installing Python 3.11 via uv..."
if which python3.11 &>/dev/null; then
    echo "  Python 3.11 already available: $(which python3.11)"
else
    uv python install 3.11
    echo "  Python 3.11 installed: $(uv find python3.11)"
fi

# ── 3. Create uv venv and link Python 3.11 ──────────────────────────────────
#echo ""
#UV_ENV="/workspace/uv_env/rag"
#echo "[3/4] Creating uv venv at ${UV_ENV}..."
#if [ -f "${UV_ENV}/bin/activate" ]; then
#    echo "  venv already exists at ${UV_ENV}"
#else
#    uv venv --python 3.11 "${UV_ENV}"
#    echo "  venv created at ${UV_ENV}"
#fi

#UV_PYTHON="$(which python3.11)"
#echo "  Linking ${UV_ENV}/bin/python3* -> ${UV_PYTHON}"
#ln -sf "${UV_PYTHON}" "${UV_ENV}/bin/python"
#ln -sf "${UV_PYTHON}" "${UV_ENV}/bin/python3"
#ln -sf "${UV_PYTHON}" "${UV_ENV}/bin/python3.11"
#echo "  python --version: $(${UV_ENV}/bin/python --version)"

# ── 4. Install Claude Code ──────────────────────────────────────────────────
echo ""
echo "[4/4] Installing Claude Code..."
if command -v claude &>/dev/null; then
    echo "  Claude Code already installed: $(claude --version)"
else
    # Claude Code requires Node.js >= 18
    if ! command -v node &>/dev/null || [ "$(node -e 'console.log(parseInt(process.version.slice(1)))')" -lt 18 ]; then
        echo "  Installing Node.js via uv..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash - 2>/dev/null
        apt-get install -y nodejs 2>/dev/null || {
            echo "  apt not available, trying nvm..."
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
            export NVM_DIR="$HOME/.nvm"
            [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
            nvm install 22
        }
    fi
    echo "  Node.js: $(node --version)"
    npm install -g @anthropic-ai/claude-code
    echo "  Claude Code installed: $(claude --version)"
fi

# ── Update ~/.bashrc with PATH for uv and claude ────────────────────────────
BASHRC="$HOME/.bashrc"
PATH_LINE='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -qF '$HOME/.local/bin' "${BASHRC}" 2>/dev/null; then
    echo "" >> "${BASHRC}"
    echo "# uv and claude PATH" >> "${BASHRC}"
    echo "${PATH_LINE}" >> "${BASHRC}"
    echo "  Added PATH entry to ${BASHRC}"
else
    echo "  PATH entry already present in ${BASHRC}"
fi

# ── 5. Reinstall KnowledgeForge venv packages ────────────────────────────────
echo ""
KF_DIR="/workspace/knowledgeforge"
echo "[5/5] Reinstalling KnowledgeForge venv packages..."
if [ -f "${KF_DIR}/pyproject.toml" ]; then
    cd "${KF_DIR}"
    uv sync --reinstall
    echo "  KnowledgeForge packages reinstalled via uv sync"
    cd -
else
    echo "  WARNING: ${KF_DIR}/pyproject.toml not found, skipping"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Setup complete!"
echo "  uv:          $(uv --version)"
echo "  python:      $(python --version 2>&1)"
echo "  claude:      $(claude --version 2>&1)"
echo "========================================="



git config --global user.email "devadotsrini@gmail.com"
git config --global user.name "Deva"