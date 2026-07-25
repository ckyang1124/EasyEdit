#!/usr/bin/env python3
"""
Independent LLM-based evaluation script for text-based portability results.

This script reads the flat JSON list produced by text_based_portability.py
(each item: file, original_answer, edited_answer, portability_question,
portability_answer, model_response) and uses the OpenAI API to judge whether
model_response is correct with respect to portability_answer.

This version uses individual API calls for maximum compatibility with all API key types.

Usage:
    python llm_evaluator_text_based_portability.py \
        --input_file model_outputs/text_based_portability/DeSTA/single/Animal.json \
        --output_file results/evaluated_text_based_portability_DeSTA_Animal.json

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
) -> tuple[bool, str, str]:
    """Evaluate correctness of a model response using OpenAI API."""

    if model_response.strip().lower() == ground_truth.strip().lower():
        return True, "Exact match with ground truth", "correct"

    try:
        all_options = question.split("\n\nOptions:\n")[-1].split("\n")
        wrong_options = [
            opt.strip() for opt in all_options if opt.strip() != ground_truth.strip()
        ]
        if ground_truth in all_options and model_response.strip() in wrong_options:
            return False, "Model response matches a wrong option exactly", "incorrect"
    except Exception:
        pass  # Ignore option parsing errors

    prompt = f"""You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
1. Since there is one and only one correct answer, it should be judged incorrect if the model does not choose any option from the option list or it choose more than one option.
2. If the model chooses one option from the option list, it should be judged correct if the chosen option aligns with the ground truth answer, otherwise it should be judged incorrect.
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
            result_lower = result.lower()
            judgement = result_lower.split("judgement:")[-1].strip()
            if judgement == "correct":
                is_correct = True
            elif judgement == "incorrect":
                is_correct = False
            else:
                is_correct = None

            if is_correct is not None:
                return (is_correct, explanation, result)

    except Exception as e:
        tqdm.write(f"API call failed: {e}")

    # Fallback to string comparison
    fallback_correct = model_response.strip().lower() == ground_truth.strip().lower()
    return (
        fallback_correct,
        "Fallback: API call failed, used string comparison",
        "correct" if fallback_correct else "incorrect",
    )


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


def evaluate_portability_items(
    items: List[Dict],
    api_delay: float = 0.1,
    reasoning_effort: str = "low",
) -> List[Dict[str, Any]]:
    """
    Evaluate correctness for the flat list of text-based portability items.

    Each item is expected to have: file, original_answer, edited_answer,
    portability_question, portability_answer, model_response.

    Returns a list of items augmented with correctness evaluation fields.
    """
    evaluated_items = []

    for i, item in enumerate(tqdm(items, desc="Evaluating portability", dynamic_ncols=True)):
        model_response = item.get("model_response", "")
        ground_truth = item.get("portability_answer", "")
        question = item.get("portability_question", "")

        evaluated_item = dict(item)

        if not model_response or model_response == "<missing>":
            evaluated_item.update(
                {
                    "correct": False,
                    "skipped": True,
                    "explanation": "Skipped due to missing response",
                    "raw_llm_response": "",
                }
            )
            evaluated_items.append(evaluated_item)
            continue

        tqdm.write(f"  Processing item {i+1}/{len(items)}...", end="")
        is_correct, explanation, llm_response = llm_judge_correctness(
            model_response, ground_truth, question, reasoning_effort
        )
        tqdm.write(" ✓" if is_correct else " ✗")

        evaluated_item.update(
            {
                "correct": is_correct,
                "skipped": False,
                "explanation": explanation,
                "raw_llm_response": llm_response,
            }
        )
        evaluated_items.append(evaluated_item)

        # Add small delay to be respectful to API limits
        if explanation not in [
            "Exact match with ground truth",
            "Model response matches a wrong option exactly",
        ]:
            time.sleep(api_delay)

    return evaluated_items


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text-based portability correctness using LLM judge"
    )
    parser.add_argument(
        "--input_file",
        required=False,
        default="model_outputs/text_based_portability/DeSTA/single/Animal.json",
    )
    parser.add_argument(
        "--output_file",
        required=False,
        default="model_outputs/text_based_portability/DeSTA/single/eval_result/Animal.json",
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
        default=None,
        help="Maximum number of items to process (default: all items)",
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

    if not data:
        tqdm.write("No results found in input file")
        return 1

    items = data[: args.max_items] if args.max_items else data

    tqdm.write(f"Found {len(data)} portability items, evaluating {len(items)}")
    tqdm.write(f"Using model: {OPENAI_MODEL_NAME}")
    tqdm.write(
        f"Using individual OpenAI API calls with {args.api_delay}s delay between calls"
    )
    tqdm.write(f"Reasoning effort level: {args.reasoning_effort.upper()}")

    try:
        evaluated_items = evaluate_portability_items(
            items, args.api_delay, args.reasoning_effort
        )
    except KeyboardInterrupt:
        tqdm.write("Processing interrupted by user. Stopping further processing.")
        return 1

    correct_count = sum(1 for item in evaluated_items if item.get("correct"))
    total_items = len(evaluated_items)
    accuracy = correct_count / total_items if total_items else 0.0

    overall_summary = {
        "total_items": total_items,
        "correct_count": correct_count,
        "portability_acc": accuracy,
    }

    # Save evaluated results
    output_data = {
        "results": evaluated_items,
        "summary": overall_summary,
        "evaluation_metadata": {
            "evaluator": "llm_evaluator_text_based_portability.py",
            "model_used": OPENAI_MODEL_NAME,
            "api_base": "https://api.openai.com/v1/",
            "evaluates": ["portability_correctness"],
            "processing": {
                "method": "individual_api_calls",
                "api_delay_seconds": args.api_delay,
            },
        },
    }

    print(f"\nSaving evaluated results to {args.output_file}...")
    if os.path.dirname(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {OPENAI_MODEL_NAME}")
    print(f"Reasoning effort: {args.reasoning_effort.upper()}")
    print(f"Total items evaluated: {total_items}")
    print()
    print(f"Portability accuracy: {correct_count}/{total_items} correct ({accuracy:.3f})")
    print()
    print(f"Results saved to: {args.output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
