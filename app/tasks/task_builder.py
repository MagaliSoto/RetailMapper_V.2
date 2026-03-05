import logging
from typing import Dict, List, Any, Tuple, Optional, Set


logger = logging.getLogger(__name__)


class TaskBuilder:
    """
    TaskBuilder generates human-readable shelf correction tasks
    based on planogram comparison results.

    It processes:
    - Missing products
    - Misplaced products (including swap detection)
    - Unexpected products

    Output is a professionally ordered list of correction sentences.
    """

    # =====================================================
    # PUBLIC API
    # =====================================================

    def build_tasks(self, data: dict) -> List[str]:
        """
        Main entrypoint.

        Builds, deduplicates, and sorts correction tasks
        based on planogram comparison data.
        """
        logger.info("Building correction tasks")

        tasks = []
        seen = set()

        valid_products = self._extract_valid_match_products(data)
        position_map = self._build_position_map(data, valid_products)

        misplaced_tasks, swap_consumed = self._build_misplaced_tasks(data, position_map)
        tasks += misplaced_tasks

        replacement_tasks, replacement_consumed = self._build_replacement_tasks(data, position_map, swap_consumed)

        tasks += replacement_tasks

        all_consumed = swap_consumed | replacement_consumed

        tasks += self._build_missing_tasks(data, position_map, all_consumed)
        tasks += self._build_missing_stock_tasks(data)
        tasks += self._build_unexpected_tasks(data, all_consumed)

        logger.debug("Total raw tasks generated: %d", len(tasks))

        # Deduplicate by sentence
        unique_tasks = []
        for task in tasks:
            if task["sentence"] not in seen:
                unique_tasks.append(task)
                seen.add(task["sentence"])

        logger.debug("Total unique tasks after deduplication: %d", len(unique_tasks))

        # Professional sorting priority
        priority_map = {
            "retirar": 0,
            "reponer": 1,
            "mover": 2
        }

        unique_tasks.sort(
            key=lambda t: (
                t["row"],
                priority_map.get(t["action"], 99),
                t["col"]
            )
        )

        logger.info("Task building completed")

        return [t["sentence"] for t in unique_tasks]

    # =====================================================
    # TASK BUILDERS
    # =====================================================

    def _build_missing_stock_tasks(self, data: dict) -> List[Dict]:
        """
        Generates tasks for products that are present
        but with insufficient stock.
        """
        logger.debug("Processing missing stock products")

        tasks = []
        debug_data = data.get("debug", {})

        for pid in data.get("missing_stock", []):
            detections = debug_data.get(str(pid), [])

            if not detections:
                continue

            det = detections[0]

            label = det.get("label")
            row = det.get("detected", {}).get("row")
            col = det.get("detected", {}).get("adjusted_col")

            if label is None or row is None or col is None:
                continue

            sentence = f"Estante {row}: agregar más cantidad de {label}."

            tasks.append({
                "row": row,
                "col": col,
                "action": "reponer",
                "sentence": sentence
            })

        logger.debug("Missing stock tasks generated: %d", len(tasks))
        return tasks

    # -----------------------------------------------------

    def _build_missing_tasks(
        self,
        data: dict,
        position_map: dict,
        consumed_positions: Set[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Generates tasks for products that are missing
        from expected positions.
        """
        logger.debug("Processing missing products")

        tasks = []

        for item in data.get("missing", []):
            label = item.get("label")
            positions = item.get("positions", {})

            for row_str, cols in positions.items():
                row = int(row_str)

                for col in cols:
                    if (row, col) in consumed_positions:
                        continue
                    sentence = self._build_sentence(
                        row=row,
                        col=col,
                        label=label,
                        position_map=position_map,
                        action="reponer"
                    )

                    tasks.append({
                        "row": row,
                        "col": col,
                        "action": "reponer",
                        "sentence": sentence
                    })

        logger.debug("Missing tasks generated: %d", len(tasks))
        return tasks

    # -----------------------------------------------------

    def _build_misplaced_tasks(
        self,
        data: dict,
        position_map: dict
    ) -> Tuple[List[Dict], Set[Tuple[int, int]]]:
        """
        Generates tasks for misplaced products.

        Includes swap detection logic:
        If two products are in each other's expected positions,
        a single 'interchange' task is generated.
        """
        logger.debug("Processing misplaced products")

        tasks = []
        debug_data = data.get("debug", {})
        consumed_positions = set()

        # Build quick lookup structure by product ID
        id_info = {}

        for pid_str, detections in debug_data.items():
            if not detections:
                continue

            det = detections[0]

            if det.get("status") != "different_location":
                continue

            id_info[int(pid_str)] = {
                "label": det["label"],
                "detected": (
                    det["detected"]["row"],
                    det["detected"]["adjusted_col"]
                ),
                "expected": (
                    det["expected"]["row"][0],
                    det["expected"]["col"][0]
                )
            }

        visited = set()

        for pid, info in id_info.items():

            if pid in visited:
                continue

            label = info["label"]
            detected_pos = info["detected"]
            expected_pos = info["expected"]

            swap_found = False

            # Detect potential swap
            for other_pid, other_info in id_info.items():
                if other_pid == pid or other_pid in visited:
                    continue

                if (
                    expected_pos == other_info["detected"]
                    and detected_pos == other_info["expected"]
                ):
                    
                    sentence = (
                        f"Estante {expected_pos[0]}: intercambiar "
                        f"{label} con {other_info['label']}."
                    )

                    tasks.append({
                        "row": expected_pos[0],
                        "col": expected_pos[1],
                        "action": "mover",
                        "sentence": sentence
                    })
                    
                    consumed_positions.add(expected_pos)
                    consumed_positions.add(detected_pos)
                    
                    visited.add(pid)
                    visited.add(other_pid)
                    swap_found = True

                    logger.debug(
                        "Swap detected between '%s' and '%s'",
                        label,
                        other_info["label"]
                    )

                    break

            if swap_found:
                continue

            # Standard move task
            sentence = self._build_sentence(
                row=expected_pos[0],
                col=expected_pos[1],
                label=label,
                position_map=position_map,
                action="mover",
                current_row=detected_pos[0]
            )

            tasks.append({
                "row": expected_pos[0],
                "col": expected_pos[1],
                "action": "mover",
                "sentence": sentence
            })

            visited.add(pid)

        logger.debug("Misplaced tasks generated: %d", len(tasks))
        return tasks, consumed_positions

    # -----------------------------------------------------

    def _build_unexpected_tasks(
        self,
        data: dict,
        consumed_positions: Set[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Generates tasks for unexpected products
        that should be removed from shelves.
        """
        logger.debug("Processing unexpected products")

        tasks = []
        debug_data = data.get("debug", {})

        for item in data.get("unexpected", []):
            for label, ids in item.items():

                for pid in ids:
                    detections = debug_data.get(str(pid), [])
                    if not detections:
                        continue

                    row = detections[0]["detected"]["row"]
                    col = detections[0]["detected"]["adjusted_col"]

                    if (row, col) in consumed_positions:
                        continue

                    sentence = f"Estante {row}: retirar {label}."

                    tasks.append({
                        "row": row,
                        "col": col,
                        "action": "retirar",
                        "sentence": sentence
                    })

        logger.debug("Unexpected tasks generated: %d", len(tasks))
        return tasks

    # -----------------------------------------------------

    def _build_replacement_tasks(
        self,
        data: dict,
        position_map: dict,
        already_consumed: Set[Tuple[int, int]]
    ) -> Tuple[List[Dict], Set[Tuple[int, int]]]:
        """
        Detects when a product is unexpected in a position
        where another product is missing.

        Generates a single replacement task instead of
        separate remove + reponer tasks.
        """

        logger.debug("Processing replacement tasks")

        tasks = []
        consumed_positions = set()

        debug_data = data.get("debug", {})

        # Build missing lookup by (row, col)
        missing_lookup = {}

        for item in data.get("missing", []):
            label = item.get("label")
            positions = item.get("positions", {})

            for row_str, cols in positions.items():
                row = int(row_str)
                for col in cols:
                    missing_lookup[(row, col)] = label

        # Check unexpected products
        for item in data.get("unexpected", []):
            for extra_label, ids in item.items():

                for pid in ids:
                    detections = debug_data.get(str(pid), [])
                    if not detections:
                        continue

                    row = detections[0]["detected"]["row"]
                    col = detections[0]["detected"]["adjusted_col"]

                    pos = (row, col)
                    
                    if pos in already_consumed:
                        continue

                    if pos in missing_lookup:
                        expected_label = missing_lookup[pos]

                        sentence = (
                            f"Estante {row}: retirar {extra_label} "
                            f"y reponer {expected_label} en su lugar."
                        )

                        tasks.append({
                            "row": row,
                            "col": col,
                            "action": "retirar",  # prioridad más alta
                            "sentence": sentence
                        })

                        consumed_positions.add(pos)

        logger.debug("Replacement tasks generated: %d", len(tasks))
        return tasks, consumed_positions

    # =====================================================
    # HELPERS
    # =====================================================

    def _extract_valid_match_products(self, data: dict) -> Set[str]:
        """
        Extracts labels that are correctly matched.
        These are used to build spatial reference context.
        """
        valid = set()

        for item in data.get("match", []):
            for label in item.keys():
                valid.add(label)

        return valid

    # -----------------------------------------------------

    def _build_position_map(self, data: dict, valid_products: Set[str]) -> Dict[Tuple[int, int], List[str]]:
        """
        Builds a map of (row, col) -> list of valid labels
        to support spatial reference sentence generation.
        """
        position_map = {}
        debug_data = data.get("debug", {})

        for _, detections in debug_data.items():
            for det in detections:
                label = det.get("label")

                if label not in valid_products:
                    continue

                detected = det.get("detected", {})
                row = detected.get("row")
                col = detected.get("adjusted_col")

                if row is not None and col is not None:
                    position_map.setdefault((row, col), []).append(label)

        return position_map

    # -----------------------------------------------------

    def _get_single_neighbor(
        self,
        position_map: Dict[Tuple[int, int], List[str]],
        row: int,
        col: int
    ) -> Optional[str]:
        """
        Returns the single product at a given position
        if and only if exactly one exists.
        """
        labels = position_map.get((row, col))

        if not labels:
            return None

        if len(labels) == 1:
            return labels[0]

        return None

    # -----------------------------------------------------

    def _build_sentence(
        self,
        row: int,
        col: int,
        label: str,
        position_map: dict,
        action: str,
        current_row: Optional[int] = None
    ) -> str:
        """
        Builds a professional correction sentence with optional
        spatial references based on neighboring products.
        """

        left_product = self._get_single_neighbor(position_map, row, col - 1)
        right_product = self._get_single_neighbor(position_map, row, col + 1)

        if left_product == label:
            left_product = None

        if right_product == label:
            right_product = None

        # Base sentence according to action
        if action == "mover":

            if current_row is None:
                base = f"Estante {row}: mover {label} al estante {row}"

            elif current_row != row:
                base = (
                    f"Estante {row}: mover {label} "
                    f"desde estante {current_row} al estante {row}"
                )

            else:
                base = (
                    f"Estante {row}: mover {label} "
                    f"dentro del estante {row}"
                )

        else:
            base = f"Estante {row}: {action} {label}"

        # Spatial reference enrichment
        if left_product and right_product and left_product != right_product:
            return f"{base} entre {left_product} y {right_product}."

        if right_product:
            return f"{base} a la izquierda de {right_product}."

        if left_product:
            return f"{base} a la derecha de {left_product}."

        if action == "reponer":
            return f"{base} en su posición correcta."

        if action == "mover":
            return f"{base} a su posición correcta."

        return f"{base}."