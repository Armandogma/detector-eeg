"""Ubicaciones compartidas; todas se resuelven desde la carpeta del proyecto."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANALISIS = REPO / "analisis"
ARTICULOS = REPO / "articulos"
DATASET = REPO / "dataset"
INFORMES = REPO / "informes"
MEDICIONES = REPO / "mediciones"
TEXTOS = REPO / "textos"
REVISION = REPO / "revisar"
