"""Revisa que el notebook se pueda entregar: ejecutado en orden, sin errores y con salidas.
Uso: python verificar_notebook.py proyecto_similitud_fotos_v6.ipynb"""
import json, sys, os

ruta = sys.argv[1] if len(sys.argv) > 1 else "proyecto_similitud_fotos_v6.ipynb"
nb = json.load(open(ruta, encoding="utf-8"))
cod = [c for c in nb["cells"] if c["cell_type"] == "code"]
fallos = []

cont = [c.get("execution_count") for c in cod]
if None in cont:
    fallos.append(f"{cont.count(None)} celdas de código sin ejecutar")
elif cont != list(range(1, len(cod) + 1)):
    fallos.append(f"contadores no consecutivos desde 1: {cont}")

for i, c in enumerate(cod, 1):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            fallos.append(f"celda de código {i}: error {o.get('ename')}")

# Celdas que deberían producir salida (figuras, tablas o texto): avisa si alguna quedó vacía
vacias = [i for i, c in enumerate(cod, 1) if not c.get("outputs")]
print(f"Celdas de código: {len(cod)} | sin salida: {vacias} (normal en definiciones/configuración)")

if nb["cells"][-1]["cell_type"] != "markdown" or "IA generativa" not in "".join(nb["cells"][-1]["source"]):
    fallos.append("la última celda no es la declaración de uso de IA generativa")
if "Borrador a completar" in "".join(nb["cells"][-1]["source"]):
    print("AVISO: completa nombre y fecha en la declaración de IA.")

codigo = "\n".join("".join(c["source"]) for c in cod)
for patron in ("/home/", "C:\\\\", "/Users/", "/content/drive"):
    if patron in codigo:
        fallos.append(f"posible ruta absoluta en el código: {patron}")
if "SEED" not in codigo:
    fallos.append("no se encontró SEED")

mb = os.path.getsize(ruta) / 1e6
print(f"Tamaño del notebook: {mb:.1f} MB")
print("RESULTADO:", "OK, listo para entregar" if not fallos else "REVISAR")
for f in fallos:
    print(" -", f)
sys.exit(1 if fallos else 0)
