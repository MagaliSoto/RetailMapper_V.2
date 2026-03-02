# app/tasks/task_manager.py

import logging
from typing import Dict, Any, List

from app.tasks.task_builder import TaskBuilder
from app.utils.gpt_utils import call_gpt_with_text


logger = logging.getLogger(__name__)


class TaskManager:
    """
    TaskManager orchestrates task generation and builds
    the final structured compliance output.

    It coordinates:
    - Task creation
    - Compliance score calculation
    - Executive summary generation
    """

    def __init__(self) -> None:
        self.builder = TaskBuilder()
        logger.debug("TaskManager initialized")

    # =====================================================
    # PUBLIC API
    # =====================================================

    def generate_tasks_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the final structured response:

        {
            "tasks": list,
            "score": float,
            "summary": string
        }
        """
        logger.info("Generating tasks output")

        # Build correction tasks
        tasks = self.builder.build_tasks(data)
        logger.debug("Tasks generated: %d", len(tasks))

        # Calculate compliance score
        score = self._calculate_score(data)
        logger.debug("Compliance score calculated: %.2f%%", score)

        # Generate executive summary
        summary = self._generate_summary(tasks, score)

        logger.info("Task output generation completed")

        return {
            "tasks": tasks,
            "score": score,
            "summary": summary
        }

    # =====================================================
    # INTERNAL METHODS
    # =====================================================

    def _calculate_score(self, data: Dict[str, Any]) -> float:
        """
        Calculates the compliance percentage based on validation results
        stored in the debug section.

        Formula:
            (items with status "match" / total evaluated items) * 100

        Returns:
            float: Compliance percentage rounded to 2 decimals.
                   Returns 0.0 if no items were evaluated.
        """
        logger.debug("Calculating compliance score")

        debug_data = data.get("debug", {})

        total = 0
        match_count = 0

        for items in debug_data.values():
            for item in items:
                total += 1
                if item.get("status") == "match":
                    match_count += 1

        if total == 0:
            logger.warning("No evaluated items found in debug data")
            return 0.0

        score = round((match_count / total) * 100, 2)

        logger.debug(
            "Score details - total: %d, matches: %d, score: %.2f%%",
            total,
            match_count,
            score
        )

        return score

    # -----------------------------------------------------

    def _generate_summary(self, tasks: List[str], score: float) -> str:
        """
        Generates a compliance executive summary.

        Behavior:
        - If score is 100%, returns a fixed success message.
        - If no tasks exist, returns a basic compliance statement.
        - Otherwise, generates an AI-based executive summary.
        """
        logger.debug("Generating executive summary")

        if score == 100:
            logger.debug("Full compliance detected")
            return "El planograma se encuentra correctamente ejecutado."

        if not tasks:
            logger.debug("No tasks detected, returning compliance-only summary")
            return f"Nivel de cumplimiento: {score}%."

        prompt = f"""
        Analiza la siguiente lista de tareas detectadas en una revisión de planograma.

        TAREAS:
        {tasks}

        Instrucciones:
        - Redacta un único párrafo de resumen ejecutivo.
        - No uses viñetas, listas ni subtítulos.
        - No enumeres tareas.
        - No repitas literalmente los textos.
        - Explica de manera natural qué tipo de desvíos se detectaron y en qué estantes.
        - El tono debe ser profesional, claro y humano.
        """

        try:
            logger.debug("Calling GPT for executive summary generation")
            ai_summary = call_gpt_with_text(prompt)
            logger.debug("AI summary successfully generated")
            return ai_summary

        except Exception as e:
            logger.exception("AI summary generation failed: %s", str(e))

            # Safe fallback in case of AI failure
            return (
                f"Se detectaron {len(tasks)} tareas pendientes. "
                f"Nivel de cumplimiento: {score}%."
            )