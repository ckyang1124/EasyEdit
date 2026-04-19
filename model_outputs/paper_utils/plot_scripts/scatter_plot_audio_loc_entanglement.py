import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODEL_ALIAS = {
	"DeSTA": "DeSTA2.5",
	"Qwen": "Qwen2-Audio",
	"AF": "Audio Flamingo 3",
	"DeSTA2.5": "DeSTA2.5",
	"Qwen2-Audio": "Qwen2-Audio",
	"Audio Flamingo 3": "Audio Flamingo 3",
}

MODEL_ORDER = ["DeSTA2.5", "Qwen2-Audio", "Audio Flamingo 3"]
METHOD_ORDER = ["FT (LLM)", "FT (Audio)", "KE", "MEND", "UnKE", "I-IKE", "IE-IKE", "WISE"]
AUDIO_LOC_TYPE_ORDER = ["Type 1", "Type 2", "Type 4"]
MODEL_DISPLAY_NAME = {
	"DeSTA2.5": "DeSTA",
	"Qwen2-Audio": "Qwen",
	"Audio Flamingo 3": "AF",
}
ENTANGLEMENT_TYPE_ORDER = [
	"Type 1 Entanglement",
	"Type 2 Entanglement",
	"Type 4 Entanglement",
]
ENTANGLEMENT_TYPE_DISPLAY = {
	"Type 1 Entanglement": "Type 1",
	"Type 2 Entanglement": "Type 2",
	"Type 4 Entanglement": "Type 4",
}
ENTANGLEMENT_TYPE_COLOR = {
	"Type 1 Entanglement": "#2A9D8F",
	"Type 2 Entanglement": "#E76F51",
	"Type 4 Entanglement": "#7B6DCC",
}
MODEL_STYLE = {
	"DeSTA2.5": {"size": 110},
	"Qwen2-Audio": {"size": 140},
	"Audio Flamingo 3": {"size": 175},
}
MODEL_HATCH = {
	"DeSTA2.5": "//////",
	"Qwen2-Audio": "",
	"Audio Flamingo 3": "....",
}
LEGEND_FONTSIZE = 8
LEGEND_TITLE_FONTSIZE = 9
LEGEND_COLUMNSPACING = 1.0
LEGEND_METHOD_COLUMNSPACING = 1.0
LEGEND_COMMON_Y = 0.80
LEGEND_X_START = 0.07
UNIFORM_BORDER_LINEWIDTH = 1.2


def parse_args():
	parser = argparse.ArgumentParser(
		description="Scatter: Audio locality type 2 (x) vs entanglement rate (y)."
	)
	parser.add_argument(
		"--single_csv",
		type=str,
		default=os.path.join(
			os.path.dirname(__file__), "../csvs/single_editing_all.csv"
		),
		help="Path to single_editing_all.csv",
	)
	parser.add_argument(
		"--entanglement_csv",
		type=str,
		default=os.path.join(
			os.path.dirname(__file__), "../csvs/audio_loc_entanglement.csv"
		),
		help="Path to audio_loc_entanglement.csv",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=os.path.join(
			os.path.dirname(__file__), "scatter_audio_loc_entanglement.png"
		),
		help="Output figure path",
	)
	parser.add_argument(
		"--annotate",
		action="store_true",
		help="Annotate each point with model-method text",
	)
	return parser.parse_args()


def parse_float(raw_value):
	text = str(raw_value).strip()
	if not text or text.upper() == "N/A":
		return np.nan
	text = text.replace("%", "")
	try:
		return float(text)
	except ValueError:
		return np.nan


def normalize_entanglement_type(raw_value):
	text = " ".join(str(raw_value).replace("\n", " ").split())
	if not text:
		return ""
	if text.lower().startswith("type 1"):
		return "Type 1 Entanglement"
	if text.lower().startswith("type 2"):
		return "Type 2 Entanglement"
	return text


def load_audio_loc_metrics(single_csv_path):
	"""Load audio locality Type 1/2/4 for Attr.=ALL."""
	pair_to_metrics = {}

	with open(single_csv_path, "r", encoding="utf-8", newline="") as file_obj:
		rows = list(csv.reader(file_obj))

	current_model = None
	current_method = None

	# Skip the 2-row header.
	for row in rows[2:]:
		if len(row) < 13:
			continue

		model_cell = row[0].strip()
		method_cell = row[1].strip()
		attr_cell = row[2].strip()

		if model_cell:
			current_model = model_cell
		if method_cell:
			current_method = method_cell

		if attr_cell != "ALL":
			continue
		if not current_model or not current_method:
			continue

		pair_to_metrics[(current_model, current_method)] = {
			"Type 1": parse_float(row[9]),
			"Type 2": parse_float(row[10]),
			"Type 4": parse_float(row[12]),
		}

	return pair_to_metrics


def build_audio_loc_type_map(audio_loc_metrics_map, audio_loc_type):
	"""Extract a single audio locality type map for plotting/filtering."""
	pair_to_value = {}
	for pair, metric_map in audio_loc_metrics_map.items():
		value = metric_map.get(audio_loc_type, np.nan)
		if np.isnan(value):
			continue
		pair_to_value[pair] = value

	return pair_to_value


def load_entanglement(entanglement_csv_path):
	"""Load y-axis values from audio_loc_entanglement.csv (type 1 and type 2)."""
	pair_to_value = {}

	with open(entanglement_csv_path, "r", encoding="utf-8", newline="") as file_obj:
		rows = list(csv.reader(file_obj))

	if not rows:
		return pair_to_value

	method_by_col = {}
	for col_idx, method_name in enumerate(rows[0][2:], start=2):
		method_name = method_name.strip()
		if method_name:
			method_by_col[col_idx] = method_name

	current_type = ""
	for row in rows[1:]:
		if len(row) < 2:
			continue

		type_cell = normalize_entanglement_type(row[0])
		if type_cell:
			current_type = type_cell

		if not current_type:
			continue

		raw_model = row[1].strip()
		if not raw_model:
			continue

		model_name = MODEL_ALIAS.get(raw_model, raw_model)
		for col_idx, method_name in method_by_col.items():
			if col_idx >= len(row):
				continue

			entanglement = parse_float(row[col_idx])
			if np.isnan(entanglement):
				continue

			pair_to_value[(model_name, method_name, current_type)] = entanglement

	return pair_to_value


def get_entanglement_types(entanglement_map):
	return ordered_unique(
		[pair[2] for pair in entanglement_map.keys()], ENTANGLEMENT_TYPE_ORDER
	)


def build_points(audio_loc_type2_map, entanglement_map, entanglement_types):
	points = []
	for (model_name, method_name), x_value in audio_loc_type2_map.items():
		for ent_type in entanglement_types:
			pair = (model_name, method_name, ent_type)
			if pair not in entanglement_map:
				continue
			points.append(
				{
					"model": model_name,
					"method": method_name,
					"x": x_value,
					"y": entanglement_map[pair],
					"ent_type": ent_type,
				}
			)
	return points


def ordered_unique(values, preferred_order):
	existing = set(values)
	ordered = [name for name in preferred_order if name in existing]
	extras = sorted(existing.difference(ordered))
	return ordered + extras


def format_metric(value):
	try:
		if value is None or np.isnan(value):
			return "N/A"
	except TypeError:
		return str(value)
	return f"{float(value):.2f}"


def print_raw_metrics_table(audio_loc_metrics_map, entanglement_map, entanglement_types):
	audio_loc_pairs = set(audio_loc_metrics_map.keys())
	entanglement_pairs = {(pair[0], pair[1]) for pair in entanglement_map.keys()}
	all_model_method_pairs = audio_loc_pairs | entanglement_pairs

	# Keep a stable, readable order while still supporting unexpected models/methods.
	model_names = ordered_unique(
		MODEL_ORDER + [pair[0] for pair in all_model_method_pairs], MODEL_ORDER
	)
	method_names = ordered_unique(
		METHOD_ORDER + [pair[1] for pair in all_model_method_pairs], METHOD_ORDER
	)

	audio_loc_headers = [
		f"Audio Loc {audio_loc_type} (ALL Attr., %)"
		for audio_loc_type in AUDIO_LOC_TYPE_ORDER
	]
	type_headers = [
		f"{ENTANGLEMENT_TYPE_DISPLAY.get(ent_type, ent_type)} Entanglement (%)"
		for ent_type in entanglement_types
	]

	print("\nRaw metrics for every model-method pair:")
	print(
		"Model,Method,"
		+ ",".join(audio_loc_headers)
		+ ","
		+ ",".join(type_headers)
	)

	for model_name in model_names:
		for method_name in method_names:
			pair_2d = (model_name, method_name)
			audio_loc_metric_map = audio_loc_metrics_map.get(pair_2d, {})
			audio_loc_values = [
				format_metric(audio_loc_metric_map.get(audio_loc_type, np.nan))
				for audio_loc_type in AUDIO_LOC_TYPE_ORDER
			]
			entanglement_values = [
				format_metric(
					entanglement_map.get((model_name, method_name, ent_type), np.nan)
				)
				for ent_type in entanglement_types
			]
			print(
				f"{model_name},{method_name},"
				+ ",".join(audio_loc_values)
				+ ","
				+ ",".join(entanglement_values)
			)


def render_scatter(points, output_path, entanglement_types, annotate=False):
	if not points:
		raise ValueError("No overlapping model-method pairs found between the two CSVs.")

	plt.rcParams.update({"font.size": 12})

	models = ordered_unique([point["model"] for point in points], MODEL_ORDER)
	methods = ordered_unique([point["method"] for point in points], METHOD_ORDER)
	ent_types = ordered_unique(entanglement_types, ENTANGLEMENT_TYPE_ORDER)

	marker_values = ["o", "s", "^", "D", "v", "P", "X", "*"]

	default_style = {"size": 130}
	model_to_style = {
		model_name: MODEL_STYLE.get(model_name, default_style) for model_name in models
	}
	fallback_hatches = ["///", "xx", "++", "..", "\\\\", "oo", "--", "||", "**"]
	used_hatches = {
		MODEL_HATCH[model_name]
		for model_name in models
		if model_name in MODEL_HATCH and MODEL_HATCH[model_name]
	}
	hatch_pool = [hatch for hatch in fallback_hatches if hatch not in used_hatches]
	model_to_hatch = {}
	for model_idx, model_name in enumerate(models):
		if model_name in MODEL_HATCH:
			model_to_hatch[model_name] = MODEL_HATCH[model_name]
			continue
		model_to_hatch[model_name] = hatch_pool[model_idx % len(hatch_pool)]

	type_palette_fallback = plt.get_cmap("Set2")(np.linspace(0.1, 0.9, len(ent_types)))
	type_to_color = {}
	for idx, ent_type in enumerate(ent_types):
		type_to_color[ent_type] = ENTANGLEMENT_TYPE_COLOR.get(
			ent_type, type_palette_fallback[idx]
		)

	method_to_marker = {
		method_name: marker_values[index % len(marker_values)]
		for index, method_name in enumerate(methods)
	}

	fig, axis = plt.subplots(figsize=(7, 7))

	for point in points:
		style = model_to_style[point["model"]]
		axis.scatter(
			point["x"],
			point["y"],
			color=type_to_color[point["ent_type"]],
			marker=method_to_marker[point["method"]],
			s=style["size"],
			hatch=model_to_hatch[point["model"]],
			edgecolors="black",
			linewidths=UNIFORM_BORDER_LINEWIDTH,
			alpha=0.95,
		)

		if annotate:
			short_model = MODEL_ALIAS.get(point["model"], point["model"])
			short_type = ENTANGLEMENT_TYPE_DISPLAY.get(point["ent_type"], point["ent_type"])
			axis.annotate(
				f"{short_model}-{point['method']}-{short_type}",
				(point["x"], point["y"]),
				textcoords="offset points",
				xytext=(5, 4),
				fontsize=8,
			)

	axis.set_xlabel("Audio Locality Score (%)", labelpad=2)
	axis.set_ylabel("Entanglement Rate (%)", labelpad=-3)
	# axis.set_title("Audio Locality Entanglement")
	axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

	y_values = np.array([point["y"] for point in points])

	y_min, y_max = np.min(y_values), np.max(y_values)

	axis.set_xlim(0.0, 100.0)
	axis.set_ylim(max(0.0, y_min - 3), min(100.0, y_max + 3))

	type_handles = [
		Line2D(
			[],
			[],
			linestyle="",
			marker="o",
			markersize=9,
			markerfacecolor=type_to_color[ent_type],
			markeredgecolor="black",
			label=ENTANGLEMENT_TYPE_DISPLAY.get(ent_type, ent_type),
		)
		for ent_type in ent_types
	]

	model_handles = [
		Patch(
			facecolor="white",
			edgecolor="black",
			linewidth=UNIFORM_BORDER_LINEWIDTH,
			hatch=model_to_hatch[model_name],
			label=MODEL_DISPLAY_NAME.get(model_name, model_name),
		)
		for model_name in models
	]

	method_handles = [
		Line2D(
			[],
			[],
			linestyle="",
			marker=method_to_marker[method_name],
			markersize=8,
			markerfacecolor="white",
			markeredgecolor="black",
			label=method_name,
		)
		for method_name in methods
	]

	fig.legend(
		handles=type_handles,
		title="Locality Type (Color)",
		loc="lower left",
		bbox_to_anchor=(LEGEND_X_START, LEGEND_COMMON_Y),
		frameon=True,
		ncol=2, #max(1, len(type_handles)),
		columnspacing=LEGEND_COLUMNSPACING,
		fontsize=LEGEND_FONTSIZE,
		title_fontsize=LEGEND_TITLE_FONTSIZE,
	)

	fig.legend(
		handles=model_handles,
		title="Model (Texture)",
		loc="lower left",
		bbox_to_anchor=(LEGEND_X_START + 0.24, LEGEND_COMMON_Y),
		frameon=True,
		ncol=2,#max(1, len(model_handles)),
		columnspacing=LEGEND_COLUMNSPACING,
		fontsize=LEGEND_FONTSIZE,
		title_fontsize=LEGEND_TITLE_FONTSIZE,
	)

	fig.legend(
		handles=method_handles,
		title="Method (Marker)",
		loc="lower left",
		bbox_to_anchor=(LEGEND_X_START + 0.448, LEGEND_COMMON_Y),
		ncol=4,
		columnspacing=LEGEND_METHOD_COLUMNSPACING,
		frameon=True,
		fontsize=LEGEND_FONTSIZE,
		title_fontsize=LEGEND_TITLE_FONTSIZE,
	)

	fig.subplots_adjust(left=0.10, right=0.98, top=0.78, bottom=0.12)

	output_dir = os.path.dirname(os.path.abspath(output_path))
	if output_dir and not os.path.exists(output_dir):
		os.makedirs(output_dir, exist_ok=True)

	fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
	plt.close(fig)


def main():
	args = parse_args()

	audio_loc_metrics_map = load_audio_loc_metrics(args.single_csv)
	audio_loc_type2_map = build_audio_loc_type_map(audio_loc_metrics_map, "Type 2")
	audio_loc_type1_map = build_audio_loc_type_map(audio_loc_metrics_map, "Type 1")
	audio_loc_type4_map = build_audio_loc_type_map(audio_loc_metrics_map, "Type 4")
	entanglement_map = load_entanglement(args.entanglement_csv)
	entanglement_types = get_entanglement_types(entanglement_map)
	print_raw_metrics_table(audio_loc_metrics_map, entanglement_map, entanglement_types)

	points = build_points(audio_loc_type2_map, entanglement_map, entanglement_types)
	render_scatter(points, args.output, entanglement_types, annotate=args.annotate)

	print(f"Loaded Audio Loc. Type 1 pairs: {len(audio_loc_type1_map)}")
	print(f"Loaded Audio Loc. Type 2 pairs: {len(audio_loc_type2_map)}")
	print(f"Loaded Audio Loc. Type 4 pairs: {len(audio_loc_type4_map)}")
	print(f"Loaded entanglement pairs (with type): {len(entanglement_map)}")
	print(
		"Entanglement types in plot: "
		+ ", ".join(ENTANGLEMENT_TYPE_DISPLAY.get(t, t) for t in entanglement_types)
	)
	print(f"Plotted overlapping points: {len(points)}")
	print(f"Saved figure to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
	main()



