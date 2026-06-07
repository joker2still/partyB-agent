def check_missing_slots(task_template: dict, slots: dict) -> list[str]:
    required_slots = task_template.get("required_slots", []) if task_template else []
    safe_slots = slots if isinstance(slots, dict) else {}

    missing_slots: list[str] = []
    for slot_name in required_slots:
        value = safe_slots.get(slot_name)
        if not isinstance(value, str) or not value.strip():
            missing_slots.append(slot_name)

    return missing_slots
