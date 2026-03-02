# app/tests/test_task_builder_with_planogram.py

import json
from pathlib import Path

from app.tasks.task_builder import TaskBuilder


def test_task_builder_with_planogram_json():
    """
    Test del TaskBuilder usando el archivo real
    output/planogram_compare.json
    """

    # Ruta al archivo
    json_path = Path("output\compare_planogram.json")

    if not json_path.exists():
        print(f"❌ No se encontró el archivo: {json_path}")
        return

    # Cargar JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Instanciar builder
    builder = TaskBuilder()

    # Generar tareas (asegurate que el builder use data["missing"])
    tasks = builder.build_tasks(data)

    print("\n=========== TAREAS GENERADAS ===========\n")

    if not tasks:
        print("✅ No se generaron tareas.")
        return

    print("\n".join(tasks))


if __name__ == "__main__":
    test_task_builder_with_planogram_json()