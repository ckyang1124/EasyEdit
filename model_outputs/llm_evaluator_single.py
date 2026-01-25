#!/usr/bin/env python3
"""
Independent LLM-based evaluation script for both correctness and locality consistency.

This script reads the JSON output from knowledge editing experiments and uses OpenAI API
to evaluate:
1. Correctness of model predictions against ground truth answers
2. Consistency between original and edited outputs for locality preservation

This version uses individual API calls for maximum compatibility with all API key types.

Usage:
    python llm_evaluator.py --input_file results/edit_results_*.json --output_file evaluated_results.json

Requirements:
    pip install openai python-dotenv tqdm
    export OPENAI_API_KEY="your_api_key_here"
"""

import json
import argparse
import time
from typing import List, Dict, Any
import openai
from tqdm import tqdm
import dotenv
import os

dotenv.load_dotenv()  # Load environment variables from .env file if present

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = (
    "gpt-5-mini-2025-08-07"  # Can also use "gpt-3.5-turbo" for lower cost
)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def llm_judge_correctness(
    model_response: str,
    ground_truth: str,
    question: str = "",
    reasoning_effort: str = "low",
) -> tuple[bool, str]:
    """Evaluate correctness of a model response using OpenAI API."""

    if model_response.strip().lower() == ground_truth.strip().lower():
        return True, "Exact match with ground truth"

    try:
        all_options = question.split("\n\nOptions:\n")[-1].split("\n")
        wrong_options = [
            opt.strip() for opt in all_options if opt.strip() != ground_truth.strip()
        ]
        if ground_truth in all_options and model_response.strip() in wrong_options:
            return False, "Model response matches a wrong option exactly"
    except Exception:
        pass  # Ignore option parsing errors

    prompt = f"""You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
1. Since there is one and only one correct answer, it should be judged incorrect if the model do not choose any option from the option list or it choose more than one option.
2. If the model choose one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
3. Read the question, options, ground truth answer and model generated response carefully before making a decision.

Considering the following examples:
Question: What is the capital of France?\n\nOptions:\nParis\nLondon\nBerlin\nMadrid.
Ground truth answer: Paris
If the model generated response is: "The capital of France is Tokyo.", it should be judged incorrect since it does not choose any option from the option list.
If the model generated response is: "The capital of France is Paris and London.", it should be judged incorrect since it chooses more than one option from the option list.
If the model generated response is: "The capital of France is London.", it should be judged incorrect since it chooses one option from the option list but the chosen option does not align with the ground truth answer.
If the model generated response is: "The capital of France is Paris.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
Another Question: What is the underlying emotion of the speaker?\n\nOptions:\nHappy\nSad\nAngry\nNeutral
Ground truth answer: Happy
If the model generated response is: "The speaker is happy.", it should be judged correct since it chooses one option from the option list and the chosen option aligns with the ground truth answer.
If the model generated response is: "The speaker expresses happiness.", it should be judged correct since "happiness" aligns with the ground truth answer "happy", and they are just different part of speech of the same word.
If the model generated response is: "Happiness," it should be judged correct since it is also a valid derivative of the ground truth answer "happy".

Now here is the question and the model generated response for you to judge:
Question: {question}
Ground truth answer: {ground_truth}
Model generated response: {model_response}

Carefully make your decision based on the above criteria. Return your judgement with the following format:
Explanation: <Your explanation on your judgement>
Judgement: <Your judgement, either "correct" or "incorrect">
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
            seed=0,
        )

        if response and response.choices and len(response.choices) > 0:
            result = response.choices[0].message.content.strip()

            # Parse the explanation and judgment
            explanation = ""
            is_correct = None

            # Extract explanation
            if "Explanation:" in result:
                expl_start = result.find("Explanation:") + len("Explanation:")
                expl_end = result.find("Judgement:")
                if expl_end != -1:
                    explanation = result[expl_start:expl_end].strip()
                else:
                    explanation = result[expl_start:].strip()

            # Extract judgment
            if (
                "correct" in result.lower()
                and "incorrect" not in result.lower()
                or "matches" in result.lower()
            ):
                is_correct = True
            elif "incorrect" in result.lower():
                is_correct = False

            if is_correct is not None:
                return is_correct, explanation

    except Exception as e:
        tqdm.write(f"API call failed: {e}")

    # Fallback to string comparison
    fallback_correct = model_response.strip().lower() == ground_truth.strip().lower()
    return fallback_correct, "Fallback: API call failed, used string comparison"


def llm_judge_consistency(
    original_output: str,
    new_output: str,
    question: str = "",
    reasoning_effort: str = "low",
) -> tuple[bool, str]:
    """Evaluate consistency between two model outputs using OpenAI API."""

    if original_output.strip().lower() == new_output.strip().lower():
        return True, "Outputs are identical"

    try:
        all_options = question.split("\n\nOptions:\n")[-1].split("\n")
        if (
            original_output.strip() in all_options
            and new_output.strip() in all_options
            and original_output.strip() != new_output.strip()
        ):
            return False, "Outputs differ and both match different options"
    except Exception:
        pass  # Ignore option parsing errors

    context_part = f"Question: {question}\\n\\n" if question.strip() else ""

    prompt = f"""You are an expert evaluator tasked with determining whether two model outputs are semantically consistent for the same input.

{context_part}Original Output: {original_output}

New Output: {new_output}

Task: Determine if these two outputs are semantically consistent. 

Two outputs are considered consistent if they are equivalent in their main message, even if they differ in wording, phrasing, or level of detail.

Two outputs are considered inconsistent if they provide contradictory information, reach different conclusions, or give fundamentally different answers.

Carefully make your decision based on the above criteria. Return your judgement with the following format:
Explanation: <Your explanation on your judgement>
Judgement: <Your judgement, either "consistent" or "inconsistent">
"""

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
            seed=0,
        )

        if response and response.choices and len(response.choices) > 0:
            result = response.choices[0].message.content.strip()

            # Parse the explanation and judgment
            explanation = ""
            is_consistent = None

            # Extract explanation
            if "Explanation:" in result:
                expl_start = result.find("Explanation:") + len("Explanation:")
                expl_end = result.find("Judgement:")
                if expl_end != -1:
                    explanation = result[expl_start:expl_end].strip()
                else:
                    explanation = result[expl_start:].strip()

            # Extract judgment
            result_upper = result.upper()
            if "CONSISTENT" in result_upper and "INCONSISTENT" not in result_upper:
                is_consistent = True
            elif "INCONSISTENT" in result_upper:
                is_consistent = False

            if is_consistent is not None:
                return is_consistent, explanation

    except Exception as e:
        tqdm.write(f"API call failed: {e}")

    # Fallback to string comparison
    fallback_consistent = original_output.strip().lower() == new_output.strip().lower()
    return fallback_consistent, "Fallback: API call failed, used string comparison"


def test_openai_api(reasoning_effort: str = "low") -> bool:
    """Test if the OpenAI API is working correctly."""
    if not OPENAI_API_KEY:
        tqdm.write("No API key available for testing")
        return False

    try:
        # Test with a simple chat completion
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "user", "content": "Respond with exactly one word: TEST"}
            ],
            reasoning_effort=reasoning_effort,
            seed=0,
        )

        if response and response.choices and len(response.choices) > 0:
            result = response.choices[0].message.content.strip()
            tqdm.write(f"✓ OpenAI API test successful. Response: {result}")
            return True
        else:
            tqdm.write("✗ OpenAI API test failed: No response content")
            return False

    except Exception as e:
        tqdm.write(f"✗ OpenAI API test failed with error: {e}")
        return False


def evaluate_correctness_batch(
    responses: List[Dict],
    output_type: str = "responses",
    api_delay: float = 0.1,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """
    Evaluate correctness for a batch of responses using individual OpenAI API calls.

    Args:
        responses: List of dicts with 'response', 'ground_truth', 'question'
        output_type: Type of output for logging purposes

    Returns:
        Dict with evaluation results
    """
    if not responses:
        return {
            "total_items": 0,
            "correct_count": 0,
            "accuracy": 0.0,
            "evaluations": [],
        }

    tqdm.write(f"\nEvaluating {len(responses)} {output_type} for correctness...")

    evaluations = []
    correct_count = 0

    for i, item in enumerate(responses):
        response = item.get("response", "")
        ground_truth = item.get("ground_truth", "")
        question = item.get("question", "")

        if not response or response == "<missing>":
            # Skipped item
            evaluation = {
                "model_response": response,
                "ground_truth": ground_truth,
                "question": question,
                "correct": False,
                "skipped": True,
                "reason": "Missing response",
                "explanation": "Skipped due to missing response",
            }
            evaluations.append(evaluation)
            continue

        tqdm.write(f"  Processing item {i+1}/{len(responses)}...", end="")
        is_correct, explanation = llm_judge_correctness(
            response, ground_truth, question, reasoning_effort
        )
        tqdm.write(" ✓" if is_correct else " ✗")

        evaluation = {
            "model_response": response,
            "ground_truth": ground_truth,
            "question": question,
            "correct": is_correct,
            "skipped": False,
            "explanation": explanation,
        }
        evaluations.append(evaluation)

        if is_correct:
            correct_count += 1

        # Add small delay to be respectful to API limits
        if explanation not in [
            "Exact match with ground truth",
            "Model response matches a wrong option exactly",
        ]:
            time.sleep(api_delay)

    accuracy = correct_count / len(responses) if responses else 0.0

    return {
        "total_items": len(responses),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "evaluations": evaluations,
    }


def evaluate_locality_outputs(
    responses: List[Dict],
    output_type: str = "audio",
    api_delay: float = 0.1,
    reasoning_effort: str = "low",
) -> Dict[str, Any]:
    """
    Evaluate a list of locality outputs using individual OpenAI API calls, comparing original outputs with new outputs.

    Args:
        responses: List of dicts with 'response', 'original_response', 'question', 'audio_file' (if audio)
        output_type: Either "audio" or "text" for logging purposes

    Returns:
        Dict with evaluation results
    """

    tqdm.write(
        f"\nEvaluating {len(responses)} {output_type} locality items for consistency..."
    )

    evaluations = []
    consistent_count = 0

    for i, item in enumerate(responses):
        new = item["response"]
        question = item["question"]
        original = item["original_response"]
        file_name = item["audio_file"] if output_type == "audio" else "N/A"

        tqdm.write(f"  Processing item {i+1}/{len(responses)}...", end="")
        is_consistent, explanation = llm_judge_consistency(
            original, new, question, reasoning_effort
        )
        tqdm.write(" ✓" if is_consistent else " ✗")

        evaluation = {
            "new_output": new,
            "original_output": original,
            "question": question,
            "file": file_name if output_type == "audio" else None,
            "consistent": is_consistent,
            "skipped": False,
            "explanation": explanation,
        }
        evaluations.append(evaluation)

        if is_consistent:
            consistent_count += 1

        # Add small delay to be respectful to API limits
        if explanation not in [
            "Outputs are identical",
            "Outputs differ and both match different options",
        ]:
            time.sleep(api_delay)

    consistency_rate = consistent_count / len(responses) if responses else 0.0

    return {
        "total_items": len(responses),
        "consistent_count": consistent_count,
        "consistency_rate": consistency_rate,
        "evaluations": evaluations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate locality consistency and correctness using LLM judge"
    )
    parser.add_argument(
        "--input_file",
        required=False,
        default="results/edit_results_20250829_181434.json",
    )
    parser.add_argument(
        "--output_file",
        required=False,
        default="results/evaluated_results_desta_Animal_all_test2.json",
    )
    parser.add_argument(
        "--test_api",
        action="store_true",
        default=False,
        help="Test API connectivity before evaluation",
    )
    parser.add_argument(
        "--api_delay",
        type=float,
        default=0.1,
        help="Seconds to wait between individual API calls (default: 0.1)",
    )
    parser.add_argument(
        "--reasoning_effort",
        choices=["minimal", "low", "medium", "high"],
        default="minimal",
        help="OpenAI API reasoning effort level (default: minimal)",
    )
    parser.add_argument(
        "--max_items",
        type=int,
        default=300,
        help="Maximum number of items to process (default: 300 for a full track)",
    )

    args = parser.parse_args()

    # Test API if requested
    if args.test_api:
        tqdm.write("Testing OpenAI API...")
        if not test_openai_api(args.reasoning_effort):
            tqdm.write("API test failed. Please check your configuration.")
            return 1
        tqdm.write("")

    # Load input results
    tqdm.write(f"Loading results from {args.input_file}...")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        tqdm.write(f"Error: Input file {args.input_file} not found")
        return 1
    except json.JSONDecodeError as e:
        tqdm.write(f"Error: Invalid JSON in {args.input_file}: {e}")
        return 1

    results = data
    if not results:
        tqdm.write("No results found in input file")
        return 1

    tqdm.write(f"Found {len(results)} edit results to evaluate")
    tqdm.write(f"Testing with first {args.max_items} items only")
    tqdm.write(f"Using model: {OPENAI_MODEL_NAME}")
    tqdm.write(
        f"Using individual OpenAI API calls with {args.api_delay}s delay between calls"
    )
    tqdm.write(f"Reasoning effort level: {args.reasoning_effort.upper()}")

    # Process each edit result
    evaluated_results = []
    total_audio_items = 0
    total_text_items = 0
    total_audio_consistent = 0
    total_text_consistent = 0

    # Correctness tracking
    total_reliability_correct = 0
    total_generality_correct = 0
    total_portability_correct = 0
    total_reliability_items = 0
    total_generality_items = 0
    total_portability_items = 0

    # Process only up to the minimum of both lengths, limited by max_items argument
    max_items = min(len(results), args.max_items)
    try:
        for i in tqdm(range(max_items), desc="Processing edits", dynamic_ncols=True):
            result = results[i]
            audio_file = (
                result["reliability"]["audio_path"].split("/")[-1]
                if result.get("reliability") and result["reliability"].get("audio_path")
                else "unknown"
            )
            tqdm.write(f"\n=== Processing edit {i+1}/{max_items}: {audio_file} ===")

            # Evaluate reliability correctness
            reliability_correct = 0
            reliability_items = 0
            reliability_eval = {}
            reliability_response = [
                {
                    "response": result["post_edit"]["reliability"],
                    "ground_truth": result["reliability"]["answer"],
                    "question": result["reliability"]["question"],
                }
            ]
            reliability_eval = evaluate_correctness_batch(
                reliability_response,
                "reliability",
                args.api_delay,
                args.reasoning_effort,
            )
            reliability_correct = reliability_eval["correct_count"]
            reliability_items = reliability_eval["total_items"]
            total_reliability_correct += reliability_correct
            total_reliability_items += reliability_items

            # Evaluate generality correctness
            generality_correct = 0
            generality_items = 0
            generality_eval = {}
            generality_responses = []
            for key in [k for k in result.keys() if k.startswith("generality")]:
                generality_responses.append(
                    {
                        "response": result["post_edit"][key],
                        "ground_truth": result[key]["answer"],
                        "question": result[key]["question"],
                    }
                )
            generality_eval = evaluate_correctness_batch(
                generality_responses,
                "generality",
                args.api_delay,
                args.reasoning_effort,
            )
            generality_correct = generality_eval["correct_count"]
            generality_items = generality_eval["total_items"]
            total_generality_correct += generality_correct
            total_generality_items += generality_items

            # Evaluate portability correctness
            portability_correct = 0
            portability_items = 0
            portability_eval = {}
            portability_response = [
                {
                    "response": result["post_edit"]["portability_audio"],
                    "ground_truth": result["portability_audio"]["answer"],
                    "question": result["portability_audio"]["question"],
                }
            ]

            if (
                result["post_edit"]["portability_audio"].strip().lower()
                == result["reliability"]["answer"].strip().lower()
            ) and (
                result["post_edit"]["portability_audio"].strip().lower()
                != result["portability_audio"]["answer"].strip().lower()
            ):
                tqdm.write(f"\nEvaluating 1 portability for correctness...")
                portability_eval = {
                    "total_items": 1,
                    "correct_count": 0,
                    "accuracy": 0.0,
                    "evaluations": [
                        {
                            "model_response": result["post_edit"]["portability_audio"],
                            "ground_truth": result["portability_audio"]["answer"],
                            "question": result["portability_audio"]["question"],
                            "correct": False,
                            "skipped": False,
                            "explanation": "Portability response exactly matches reliability answer but not portability ground truth",
                        }
                    ],
                }
                tqdm.write(f"  Processing item 1/1... ✗")

            else:
                portability_eval = evaluate_correctness_batch(
                    portability_response,
                    "portability",
                    args.api_delay,
                    args.reasoning_effort,
                )
            portability_correct = portability_eval["correct_count"]
            portability_items = portability_eval["total_items"]
            total_portability_correct += portability_correct
            total_portability_items += portability_items

            # Evaluate audio locality consistency
            audio_locality_responses = []
            for key in [k for k in result.keys() if k.startswith("locality_audio")]:
                audio_locality_responses.append(
                    {
                        "response": result["post_edit"][key],
                        "original_response": result["pre_edit"][key],
                        "question": result[key]["question"],
                        "audio_file": result[key]["audio_path"].split("/")[-1],
                    }
                )
            audio_eval = evaluate_locality_outputs(
                audio_locality_responses, "audio", args.api_delay, args.reasoning_effort
            )
            total_audio_items += audio_eval["total_items"]
            total_audio_consistent += audio_eval["consistent_count"]

            # Evaluate text locality consistency
            text_locality_response = [
                {
                    "response": result["post_edit"]["locality_text"],
                    "original_response": result["pre_edit"]["locality_text"],
                    "question": result["locality_text"]["question"],
                    "audio_file": None,
                }
            ]
            text_eval = evaluate_locality_outputs(
                text_locality_response, "text", args.api_delay, args.reasoning_effort
            )
            total_text_items += text_eval["total_items"]
            total_text_consistent += text_eval["consistent_count"]

            # Create evaluated result
            evaluated_result = result.copy()
            evaluated_result.update(
                {
                    "locality_audio_evaluation": audio_eval,
                    "locality_text_evaluation": text_eval,
                    "locality_audio_acc": audio_eval["consistency_rate"],
                    "locality_text_acc": text_eval["consistency_rate"],
                    "reliability_evaluation": reliability_eval,
                    "generality_evaluation": generality_eval,
                    "portability_evaluation": portability_eval,
                    "reliability_acc": reliability_eval.get("accuracy", 0.0),
                    "generality_acc": generality_eval.get("accuracy", 0.0),
                    "portability_acc": portability_eval.get("accuracy", 0.0),
                }
            )

            evaluated_results.append(evaluated_result)

            tqdm.write(
                f"  Reliability: {reliability_correct}/{reliability_items} correct ({reliability_eval.get('accuracy', 0.0):.3f})"
            )
            tqdm.write(
                f"  Generality: {generality_correct}/{generality_items} correct ({generality_eval.get('accuracy', 0.0):.3f})"
            )
            tqdm.write(
                f"  Portability: {portability_correct}/{portability_items} correct ({portability_eval.get('accuracy', 0.0):.3f})"
            )
            tqdm.write(
                f"  Audio locality: {audio_eval['consistent_count']}/{audio_eval['total_items']} consistent ({audio_eval['consistency_rate']:.3f})"
            )
            tqdm.write(
                f"  Text locality: {text_eval['consistent_count']}/{text_eval['total_items']} consistent ({text_eval['consistency_rate']:.3f})"
            )
    except Exception as e:
        tqdm.write(f"warning: Exception occurred during processing: {e}")
        tqdm.write("Stopping further processing.")
    except KeyboardInterrupt:
        tqdm.write("Processing interrupted by user. Stopping further processing.")

    # Compute overall summary
    overall_summary = {
        "locality_audio_acc": total_audio_consistent / max(1, total_audio_items),
        "locality_text_acc": total_text_consistent / max(1, total_text_items),
        "total_locality_audio_items": total_audio_items,
        "total_locality_text_items": total_text_items,
        "total_locality_audio_consistent": total_audio_consistent,
        "total_locality_text_consistent": total_text_consistent,
        "reliability_acc": total_reliability_correct / max(1, total_reliability_items),
        "generality_acc": total_generality_correct / max(1, total_generality_items),
        "portability_acc": total_portability_correct / max(1, total_portability_items),
        "total_reliability_items": total_reliability_items,
        "total_generality_items": total_generality_items,
        "total_portability_items": total_portability_items,
        "total_reliability_correct": total_reliability_correct,
        "total_generality_correct": total_generality_correct,
        "total_portability_correct": total_portability_correct,
    }

    # Save evaluated results
    output_data = {
        "results": evaluated_results,
        "summary": overall_summary,
        "evaluation_metadata": {
            "evaluator": "llm_evaluator.py",
            "model_used": OPENAI_MODEL_NAME,
            "api_base": "https://api.openai.com/v1/",
            "evaluates": ["correctness", "locality_consistency"],
            "processing": {
                "method": "individual_api_calls",
                "api_delay_seconds": args.api_delay,
            },
        },
    }

    print(f"\nSaving evaluated results to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (TEST RUN)")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL_NAME}")
    print(f"Reasoning effort: {args.reasoning_effort.upper()}")
    print(f"Total edits evaluated: {max_items} (limited for testing)")
    print()
    print("CORRECTNESS METRICS:")
    print(
        f"  Reliability: {total_reliability_correct}/{total_reliability_items} correct ({total_reliability_correct/max(1,total_reliability_items):.3f})"
    )
    print(
        f"  Generality: {total_generality_correct}/{total_generality_items} correct ({total_generality_correct/max(1,total_generality_items):.3f})"
    )
    print(
        f"  Portability: {total_portability_correct}/{total_portability_items} correct ({total_portability_correct/max(1,total_portability_items):.3f})"
    )
    print()
    print("CONSISTENCY METRICS:")
    print(
        f"  Audio locality: {total_audio_consistent}/{total_audio_items} consistent ({total_audio_consistent/max(1,total_audio_items):.3f})"
    )
    print(
        f"  Text locality: {total_text_consistent}/{total_text_items} consistent ({total_text_consistent/max(1,total_text_items):.3f})"
    )
    print()
    print(f"Results saved to: {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
