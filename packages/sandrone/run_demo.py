import json
from pathlib import Path

from sandrone.cases import all_cases
from sandrone.plot import save_layer_contact_sheet
from sandrone.render import render_scenario
from sandrone.router import generate_power_copper_shapes, validate_layout


def main() -> None:
    out_dir = Path(__file__).with_name("out")
    out_dir.mkdir(exist_ok=True)

    for scenario in all_cases():
        layout = generate_power_copper_shapes(scenario)
        report = validate_layout(scenario, layout)
        content = render_scenario(scenario, layout)
        text_path = out_dir / f"{scenario.name}.txt"
        image_path = out_dir / f"{scenario.name}.png"
        polygon_path = out_dir / f"{scenario.name}.polygons.json"
        text_path.write_text(content, encoding="utf-8")
        polygon_path.write_text(
            json.dumps(
                {
                    group_id: [
                        {
                            "layer": polygon.layer,
                            "area": polygon.area,
                            "exterior": polygon.exterior,
                            "holes": polygon.holes,
                        }
                        for polygon in polygons
                    ]
                    for group_id, polygons in layout.group_polygons.items()
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        save_layer_contact_sheet(scenario, layout, image_path)
        print(
            f"{scenario.name}: {len(report.errors)} validation errors"
            f" -> {text_path}, {image_path}, {polygon_path}"
        )
        for error in report.errors[:10]:
            print(f"  {error}")


if __name__ == "__main__":
    main()
