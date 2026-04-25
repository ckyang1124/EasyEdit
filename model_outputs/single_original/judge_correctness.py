import dotenv
import os
from tqdm import tqdm
import openai
from argparse import ArgumentParser
import json

dotenv.load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = "gpt-5-mini-2025-08-07"

PROMPT = """You will be given a question with list of possible options, a ground truth answer and a model generated response. Determine whether the model generated response is correct based on the following criteria:
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


class LLMJudger:
    def __init__(
        self, api_key: str = OPENAI_API_KEY, model_name: str = OPENAI_MODEL_NAME
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def judge_correctness(
        self,
        model_response: str,
        ground_truth: str,
        question: str = "",
    ) -> tuple[bool, str, str]:
        """Evaluate correctness of a model response using OpenAI API."""

        if model_response.strip().lower() == ground_truth.strip().lower():
            return True, "Exact match with ground truth", "correct"

        try:
            all_options = question.split("\n\nOptions:\n")[-1].split("\n")
            wrong_options = [
                opt.strip()
                for opt in all_options
                if opt.strip() != ground_truth.strip()
            ]
            if ground_truth in all_options and model_response.strip() in wrong_options:
                return (
                    False,
                    "Model response matches a wrong option exactly",
                    "incorrect",
                )
        except Exception:
            pass  # Ignore option parsing errors

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            question=question,
                            ground_truth=ground_truth,
                            model_response=model_response,
                        ),
                    }
                ],
                reasoning_effort="low",
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
        fallback_correct = (
            model_response.strip().lower() == ground_truth.strip().lower()
        )
        return (
            fallback_correct,
            "Fallback: API call failed, used string comparison",
            "correct" if fallback_correct else "incorrect",
        )


def run_judge(data: list[dict]) -> list[dict]:
    judger = LLMJudger()
    for item in tqdm(data, desc="Judging correctness", dynamic_ncols=True):
        question = item["question"]
        ground_truth = item["original_answer"]
        model_response = item["model_response"]

        is_correct, explanation, full_response = judger.judge_correctness(
            model_response=model_response,
            ground_truth=ground_truth,
            question=question,
        )

        item["judge_result"] = {
            "judgement": "correct" if is_correct else "incorrect",
            "explanation": explanation,
            "full_response": full_response,
        }
    return data


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", "-i", type=str, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output_file", "-o", type=str, help="Path to the output JSON file"
    )
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    try:
        data = run_judge(data)
    except Exception as e:
        print(f"Error occurs: {e}")
    finally:
        total_judged = correct = 0
        for item in data:
            if "judge_result" in item:
                total_judged += 1
                if item["judge_result"]["judgement"] == "correct":
                    correct += 1
        acc = correct / total_judged if total_judged > 0 else 0
        print(f"Total judged: {total_judged}, Correct: {correct}, Accuracy: {acc:.2f}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
