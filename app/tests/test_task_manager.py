# app/tests/test_task_manager_real.py

import json
from pathlib import Path

from app.tasks.task_manager import TaskManager
import logging

# =====================================================
# 🔹 LOGGER CONFIGURATION (DEBUG)
# =====================================================

def configure_logger():
    logging.basicConfig(
        level=logging.DEBUG,  # Nivel DEBUG
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Si querés que librerías externas no spameen tanto:
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)


def test_task_manager_with_real_planogram():
    """
    Real execution test for TaskManager using:

        output/compare_planogram.json

    This test:
        - Builds tasks
        - Calculates score
        - Calls OpenAI API to generate summary
        - Prints full output

    IMPORTANT:
        This test WILL call the real OpenAI API.
    """
    configure_logger()
    json_path = Path("output/compare_planogram.json")

    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        return

    # Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manager = TaskManager()

    print("\n=========== RUNNING TASK MANAGER ===========\n")

    result = manager.generate_tasks_output(data)

    print("\n=========== RESULT ===========\n")

    print("📋 Tasks:")
    if result["tasks"]:
        for task in result["tasks"]:
            print(f"- {task}")
    else:
        print("No tasks generated.")

    print("\n📊 Score:")
    print(result["score"])

    print("\n📝 Summary:")
    print(result["summary"])

    print("\n=========== END ===========\n")


if __name__ == "__main__":
    test_task_manager_with_real_planogram()