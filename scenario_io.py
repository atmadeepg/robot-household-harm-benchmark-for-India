def flatten_scenarios(scenario_data):
    flat = []
    if "subcategories" in scenario_data:
        for subcat in scenario_data["subcategories"]:
            for s in subcat["scenarios"]:
                flat.append({
                    "scenario_id": s["scenario_id"],
                    "subcategory_id": subcat["id"],
                    "subcategory_name": subcat["name"],
                    "goal": s["goal"],
                    "target_str": s["target_str"],
                    "household_actor": s.get("household_actor", "unspecified individual"),
                    "severity": s.get("severity", "medium"),
                    "persona_eligible": True,
                    "variants_per_seed": None,
                })
    elif "macro_categories" in scenario_data:
        for macro in scenario_data["macro_categories"]:
            persona_eligible = macro.get("persona_eligible", True)
            for s in macro["scenarios"]:
                flat.append({
                    "scenario_id": s["scenario_id"],
                    "subcategory_id": macro["id"],
                    "subcategory_name": macro["name"],
                    "goal": s["goal"],
                    "target_str": s["target_str"],
                    "household_actor": s.get("household_actor", "unspecified"),
                    "severity": s.get("severity", "medium"),
                    "persona_eligible": persona_eligible,
                    "variants_per_seed": macro.get("variants_per_seed"),
                })
    else:
        raise ValueError("Scenario file must have a 'subcategories' or 'macro_categories' key.")
    return flat


def flatten_scenarios_by_id(scenario_data):
    return {s["scenario_id"]: s for s in flatten_scenarios(scenario_data)}
