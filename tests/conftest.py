"""Configurações e fixtures do pytest."""

import sys
from pathlib import Path

# Adicionar src ao path para imports funcionarem
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
