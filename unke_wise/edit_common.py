"""Shared utilities for audio knowledge editing experiments.

Provides common functions for set_seed, audio path resolution, evaluation,
results writing, and the two main editing loop patterns (single / sequential).
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_DATA_BASE = "lalm-knowledge-editing/audio_data"
PRESERVATION_DATA_PATH = "editing_method/Preservation_data/preservation_data.json"
PRESERVATION_AUDIO_DIR = "editing_method/Preservation_data/audio"

CATEGORY_DIRS = ["Animal", "Emotion", "Gender", "Language"]

DYNAMIC_SUPERB_SUBDIRS = [
    "AccentClassification_AccentdbExtended",
    "AgeClassification_CommonVoiceCorpus-Test",
    "ConversationMatching_EnShortConversation",
    "DialogueActClassification_DailyTalk",
    "EnvironmentalSoundClassification_ESC50-ExteriorAndUrbanNoises",
    "EnvironmentalSoundClassification_ESC50-HumanAndNonSpeechSounds",
    "EnvironmentalSoundClassification_ESC50-InteriorAndDomesticSounds",
    "HEARMusicGenreClassification_ISMIR04",
    "HEARPercussionInstrumentsClassification_BeijingOperaPercussion",
    "HEARPercussionInstrumentsStrokeClassification_MridangamStroke",
    "HEARPercussionInstrumentsTonicClassification_MridangamTonic",
    "HowFarAreYou_3DSpeaker",
    "HumanNonSpeechSoundRecognition_Nonspeech7k-test_CommonVoice-DeltaSegment-15",
    "InstrumentClassification_Nsynth",
    "InstrumentPitchClassification_Nsynth",
    "InstrumentSourceClassification_Nsynth",
    "MARBLEGenreClassification_MTG-Genre",
    "MARBLEKeyDetection_Giantstepskey",
    "MARBLEVocalTechniqueDetection_VocalSet",
    "MusicGenreClassification_FMA",
    "PhonologicalFeatureClassification_VoxAngeles-ConsonantPlaceOfArticulation",
    "PhonologicalFeatureClassification_VoxAngeles-MannerOfArticulation",
    "PhonologicalFeatureClassification_VoxAngeles-Phone",
    "PhonologicalFeatureClassification_VoxAngeles-VowelHeight",
    "SoundEffectDetection_RemFx",
    "SpeakerCounting_LibriTTS-TestClean",
    "SpoofDetection_ASVspoof",
    "SuperbKS_SpeechCommandsV1-Test",
    "VocalSoundRecognition_VocalSound",
    "VoiceDisorderClassification_VOICED",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 2024):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_audio_path(track: str, file_name: str) -> Optional[str]:
    """Resolve audio file path, searching known directories with DynamicSuperb fallback."""
    base = AUDIO_DATA_BASE

    if track:
        cand = os.path.join(base, track, file_name)
        if os.path.exists(cand):
            return cand

    for dir_name in CATEGORY_DIRS:
        cand = os.path.join(base, dir_name, file_name)
        if os.path.exists(cand):
            return cand

    for subdir in DYNAMIC_SUPERB_SUBDIRS:
        cand = os.path.join(base, "DynamicSuperb", subdir, file_name)
        if os.path.exists(cand):
            return cand

    cand = os.path.join(base, file_name)
    return cand if os.path.exists(cand) else None


def capture_wise_state(adapter) -> Dict[str, Any]:
    """Snapshot side-memory routing statistics from the active WISE adapter."""
    if adapter is None:
        return {}
    state: Dict[str, Any] = {
        "edits_seen": int(getattr(adapter, "editing_total_cnt", 0)),
        "side_memory_slots": len(getattr(adapter, "memory_weight", [])),
        "merge_count": int(getattr(adapter, "merge_cnt", 0)),
    }
    mean_act = getattr(adapter, "editing_mean_act", None)
    if mean_act is not None:
        state["routing_threshold"] = float(mean_act.min_act())
    weight_mask = getattr(adapter, "weight_mask", None)
    if weight_mask is not None:
        state["active_mask_density"] = float(weight_mask.float().mean().item())
    if hasattr(adapter, "layer") and hasattr(adapter.layer, "weight"):
        with torch.no_grad():
            ref = adapter.original_layer.weight.to(adapter.layer.weight.device)
            drift = (adapter.layer.weight - ref).norm().item()
        state["side_memory_drift"] = float(drift)
    return state


def restore_model_weights(model, weights_copy):
    """Restore model weights from a backup (dict or callable)."""
    if weights_copy is None:
        return
    if callable(weights_copy):
        weights_copy()
        return
    from editing_method.util import nethook
    with torch.no_grad():
        for k, v in weights_copy.items():
            param = nethook.get_parameter(model, k)
            param[...] = v.to(device=param.device, dtype=param.dtype)


def cleanup_cuda(model):
    """Clear gradients and CUDA cache."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_preservation_data_raw() -> Dict:
    """Load raw preservation data JSON."""
    with open(PRESERVATION_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_text_preserve(text_raw: List[Dict]) -> List[Dict]:
    """Build text preservation list as dicts with ``question`` key."""
    return [{"question": item["question"]} for item in text_raw]


def build_audio_preserve(audio_raw: Dict, exclude_category: str = "") -> List[Dict]:
    """Build audio preservation list, excluding a specific category."""
    items = []
    for track_name, track_items in audio_raw.items():
        if exclude_category and track_name == exclude_category:
            continue
        for entry in track_items:
            path = os.path.join(PRESERVATION_AUDIO_DIR, entry["audio"])
            if os.path.exists(path):
                items.append({"question": entry["question"], "audio_path": path})
    return items


def build_audio_preserve_only(audio_raw: Dict, include_track: str) -> List[Dict]:
    """Build audio preservation list from a single track only."""
    items = []
    for entry in audio_raw.get(include_track, []):
        path = os.path.join(PRESERVATION_AUDIO_DIR, entry["audio"])
        if os.path.exists(path):
            items.append({"question": entry["question"], "audio_path": path})
    return items


def load_metadata(metadata_file: str) -> list:
    """Load metadata items from JSON, handling both list and dict formats."""
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [
            dict({"file": k}, **v) if isinstance(v, dict) else {"file": k}
            for k, v in data.items()
        ]
    return data


def load_groups_data(metadata_file: str) -> list:
    """Load sequential groups data (list of lists) from JSON."""
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level to be an array of groups for sequential edits.")
    return data


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_all(
    gen_audio_fn: Callable[[str, str], str],
    gen_text_fn: Callable[[str], str],
    item: Dict,
    audio_path: str,
) -> Dict[str, Any]:
    """Run all evaluations for a single edit item.

    Args:
        gen_audio_fn: ``(audio_path, question) -> prediction``
        gen_text_fn:  ``(question) -> prediction``
        item:         Metadata dict for the edit
        audio_path:   Path to the edit's audio file
    """
    rel_q = item.get("reliability_question", "")
    if audio_path and os.path.exists(audio_path):
        rel_pred = gen_audio_fn(audio_path, rel_q)
    else:
        rel_pred = gen_text_fn(rel_q)

    gen_items = item.get("generality", []) or []
    gen_results = _eval_generality(gen_audio_fn, gen_text_fn, gen_items)
    gen_preds = [{"file": f, "question": q, "pred": p} for f, q, p in gen_results]

    loc_audio = item.get("locality", {}).get("audio", []) or []
    loc_audio_results = _eval_locality_audio(gen_audio_fn, gen_text_fn, loc_audio)

    loc_text = item.get("locality", {}).get("text", []) or []
    loc_text_results = _eval_locality_text(gen_text_fn, loc_text)

    port_item = item.get("portability", {}).get("audio", None)
    port_pred, port_question = _eval_portability(gen_audio_fn, gen_text_fn, port_item)

    return {
        "reliability_question": rel_q,
        "reliability_pred": rel_pred,
        "generality_preds": gen_preds,
        "locality_audio_outputs": loc_audio_results,
        "locality_text_outputs": loc_text_results,
        "portability_question": port_question,
        "portability_pred": port_pred,
    }


def _eval_generality(gen_audio_fn, gen_text_fn, items):
    results = []
    for item in items:
        file_name = item.get("file", "")
        question = item.get("question", "")
        if file_name:
            apath = resolve_audio_path(item.get("track", ""), file_name)
            if not apath:
                results.append((file_name, question, "<missing>"))
                continue
            pred = gen_audio_fn(apath, question)
        else:
            pred = gen_text_fn(question)
        results.append((file_name, question, pred))
    return results


def _eval_locality_audio(gen_audio_fn, gen_text_fn, items):
    outputs = []
    for item in items:
        file_name = item.get("file", "")
        question = item.get("question", "")
        if file_name:
            apath = resolve_audio_path(item.get("track", ""), file_name)
            if not apath:
                outputs.append({"file": file_name, "new_output": "<missing>", "question": question})
                continue
            pred = gen_audio_fn(apath, question)
        else:
            pred = gen_text_fn(question)
        outputs.append({"file": file_name, "new_output": pred, "question": question})
    return outputs


def _eval_locality_text(gen_text_fn, items):
    return [
        {"new_output": gen_text_fn(item.get("question", "")), "question": item.get("question", "")}
        for item in items
    ]


def _eval_portability(gen_audio_fn, gen_text_fn, port_item):
    if not port_item or not isinstance(port_item, dict):
        return "", ""
    question = port_item.get("question", "")
    file_name = port_item.get("file", "")
    if file_name:
        apath = resolve_audio_path(port_item.get("track", ""), file_name)
        if not apath:
            return "<missing>", question
        return gen_audio_fn(apath, question), question
    if question:
        return gen_text_fn(question), question
    return "", ""


# ---------------------------------------------------------------------------
# Results Writer
# ---------------------------------------------------------------------------

class ResultsWriter:
    """Incremental JSON results writer."""

    def __init__(
        self,
        model_name: str,
        metadata_file: str,
        dataset_size_limit: int,
        extra_config: Optional[Dict] = None,
        prefix: str = "edit",
        summary_fn: Optional[Callable] = None,
    ):
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d_%H%M%S")
        self.results_path = self.results_dir / f"{prefix}_results_{self.timestamp}.json"
        self.all_results: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {
            "model_name": model_name if isinstance(model_name, str) else str(model_name),
            "metadata_file": metadata_file,
            "dataset_size_limit": dataset_size_limit,
            "timestamp_utc": self.timestamp,
        }
        if extra_config:
            self.config.update(extra_config)
        self.summary_fn = summary_fn

    def append(self, entry: Dict[str, Any]):
        self.all_results.append(entry)

    def write(self):
        summary = self.summary_fn(self.all_results) if self.summary_fn else self._default_summary()
        payload = {"config": self.config, "results": self.all_results, "summary": summary}
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _default_summary(self) -> Dict[str, Any]:
        total = len(self.all_results)
        total_audio = sum(len(r.get("locality_audio_outputs", [])) for r in self.all_results)
        total_text = sum(len(r.get("locality_text_outputs", [])) for r in self.all_results)
        return {"num_edits": total, "locality_audio_items": total_audio, "locality_text_items": total_text}


# ---------------------------------------------------------------------------
# Main loop – single edits (edit-then-restore)
# ---------------------------------------------------------------------------

def run_single_edits(
    meta_items: list,
    dataset_size_limit: int,
    category: str,
    gen_audio_fn_factory: Callable,
    gen_text_fn: Callable,
    apply_edit_fn: Callable,
    restore_fn: Callable,
    model,
    writer: ResultsWriter,
    edit_only: bool = False,
):
    """Run single-edit-with-restore loop.

    Args:
        gen_audio_fn_factory: ``(item) -> gen_audio_fn(audio_path, question) -> str``.
            Allows per-item context (e.g. DeSTA transcription).
        apply_edit_fn: ``(item, audio_path) -> weights_copy``
        restore_fn:    ``(weights_copy) -> None``
    """
    for edit_index in tqdm(range(min(len(meta_items), dataset_size_limit)), desc="Processing edits"):
        item = meta_items[edit_index]
        file_name = item.get("file", "")

        audio_path = resolve_audio_path(category, file_name)
        if not audio_path:
            audio_path = os.path.join(AUDIO_DATA_BASE, category, file_name)
        if not os.path.exists(audio_path):
            tqdm.write(f"Skipping {file_name}: audio file not found")
            continue

        start = time.time()
        weights_copy = apply_edit_fn(item, audio_path)

        if not edit_only:
            gen_audio = gen_audio_fn_factory(item)
            eval_results = evaluate_all(gen_audio, gen_text_fn, item, audio_path)
            tqdm.write(
                f"Edit {file_name}: "
                f"locality_audio={len(eval_results['locality_audio_outputs'])} | "
                f"locality_text={len(eval_results['locality_text_outputs'])} | "
                f"generality={len(eval_results['generality_preds'])}"
            )
        else:
            eval_results = {
                "reliability_question": item.get("reliability_question", ""),
                "reliability_pred": "",
                "generality_preds": [],
                "locality_audio_outputs": [],
                "locality_text_outputs": [],
                "portability_question": "",
                "portability_pred": "",
            }

        duration = time.time() - start
        tqdm.write(f"Execution took {duration:.2f} seconds")

        entry: Dict[str, Any] = {"audio_file": file_name, "duration_seconds": float(duration)}
        entry.update(eval_results)
        writer.append(entry)
        writer.write()

        restore_fn(weights_copy)
        cleanup_cuda(model)

    writer.write()


# ---------------------------------------------------------------------------
# Main loop – sequential edits (WISE groups)
# ---------------------------------------------------------------------------

def run_sequential_edits(
    groups_data: list,
    dataset_size_limit: int,
    gen_audio_fn: Callable,
    gen_text_fn: Callable,
    setup_group_fn: Callable,
    apply_edit_fn: Callable,
    reset_group_fn: Callable,
    model,
    writer: ResultsWriter,
):
    """Run sequential group editing loop (WISE sequential).

    Args:
        setup_group_fn: ``() -> (wise_editor, adapter)``
        apply_edit_fn:  ``(wise_editor, item, audio_path) -> None``
        reset_group_fn: ``(wise_editor) -> None``
    """
    cuda_available = torch.cuda.is_available()
    num_groups = min(len(groups_data), dataset_size_limit if dataset_size_limit > 0 else len(groups_data))
    wise_editor = None

    for group_idx in range(num_groups):
        group_items = groups_data[group_idx]
        if not isinstance(group_items, list):
            raise ValueError(f"Group {group_idx} should be an array of edit entries")

        group_size = len(group_items)
        tqdm.write(f"\n=== Processing Group {group_idx + 1}/{num_groups} ({group_size} edits) ===")

        if wise_editor is not None:
            reset_group_fn(wise_editor)
        wise_editor, adapter = setup_group_fn()

        for edit_idx in tqdm(range(group_size), desc=f"Group {group_idx + 1} edits"):
            item = group_items[edit_idx]
            file_name = item.get("file", "")
            audio_path = resolve_audio_path(item.get("track", ""), file_name)
            if audio_path is None:
                tqdm.write(f"Warning: Audio file {file_name} not found, skipping")
                continue

            start = time.time()
            apply_edit_fn(wise_editor, item, audio_path)

            # Cumulative evaluation: re-evaluate all edits applied so far
            cumulative_results = []
            for eval_idx in range(edit_idx + 1):
                eval_item = group_items[eval_idx]
                eval_file = eval_item.get("file", "")
                eval_audio = resolve_audio_path(eval_item.get("track", ""), eval_file)
                if eval_audio is None:
                    continue
                ev = evaluate_all(gen_audio_fn, gen_text_fn, eval_item, eval_audio)
                ev["audio_file"] = eval_file
                cumulative_results.append(ev)

            duration = time.time() - start
            wise_state = capture_wise_state(adapter)
            tqdm.write(
                f"Edit {edit_idx + 1}/{group_size} in Group {group_idx + 1}: "
                f"{file_name} (evaluated {edit_idx + 1} edits) - {duration:.2f}s"
            )

            entry = {
                "group_index": group_idx,
                "edit_index_in_group": edit_idx,
                "step_description": f"After applying edit {edit_idx + 1} in group {group_idx + 1}",
                "num_edits_applied": edit_idx + 1,
                "cumulative_evaluations": cumulative_results,
                "duration_seconds": float(duration),
                "wise_state": wise_state,
            }
            writer.append(entry)
            writer.write()

            del cumulative_results
            if cuda_available:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        tqdm.write(f"Completed Group {group_idx + 1}. Restoring main memory for next group.")
        if cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if wise_editor is not None:
        reset_group_fn(wise_editor)

    writer.write()
    tqdm.write(f"\nCompleted all {num_groups} groups. Results saved to {writer.results_path}")
