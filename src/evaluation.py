
import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_iou(dataset, s_result):
    if dataset["file_path"] != s_result["file_path"]:
        return 0.0

    dataset_start = dataset["first_character_index"]
    dataset_end = dataset["last_character_index"]
    s_result_start = s_result["first_character_index"]
    s_result_end = s_result["last_character_index"]

    intersection = max(
        0,
        min(dataset_end, s_result_end)
        - max(dataset_start, s_result_start),
    )

    union = max(dataset_end, s_result_end) - min(
        dataset_start, s_result_start
    )

    if union <= 0:
        return 0.0

    return intersection / union


def is_chunk_valid(dataset, s_result, iou_threshold):
    return calculate_iou(dataset, s_result) >= iou_threshold


def recall_number_k(expected_s, retrieved_s, k, iou_threshold):
    if not expected_s:
        return None

    k_retrieved = retrieved_s[:k]
    matched = 0
    for expected in expected_s:
        for retrieved in k_retrieved:
            if is_chunk_valid(expected, retrieved, iou_threshold):
                matched += 1
                break

    return matched / len(expected_s)


def eval(
    dataset_path,
    student_search_result_path,
    k=10,
    iou_threshold=0.05,
):
    dataset = load_json(dataset_path)
    student_data = load_json(student_search_result_path)

    questions = dataset["rag_questions"]
    student_results = student_data["search_results"]

    student_by_id = {
        result["question_id"]: result["retrieved_sources"]
        for result in student_results
    }

    evaluated_questions = []

    for question in questions:
        question_id = question["question_id"]
        expected_sources = question.get("sources", [])

        if question_id not in student_by_id:
            continue

        evaluated_questions.append(
            (question["sources"], student_by_id[question_id])
        )

    print("------------------------------")
    print(f"Dataset questions: {len(questions)}")
    print(f"Student results: {len(student_results)}")
    print(f"Questions evaluated: {len(evaluated_questions)}")
    print(f"IoU threshold: {iou_threshold}")
    print("------------------------------")

    if not evaluated_questions:
        print("No questions available for evaluation.")
        return {}

    k_values = [value for value in (1, 3, 5, 10) if value <= k]

    if k not in k_values:
        k_values.append(k)

    results = {}

    for current_k in sorted(set(k_values)):
        question_recalls = []

        for expected_sources, student_result in evaluated_questions:

            recall = recall_number_k(
                expected_sources,
                student_result,
                current_k,
                iou_threshold,
            )

            if recall is not None:
                question_recalls.append(recall)

        if question_recalls:
            average_recall = sum(question_recalls) / len(question_recalls)
            results[current_k] = average_recall

    print("\nEvaluation Results")
    print("========================================")

    for current_k, recall in results.items():
        print(
            f"Recall@{current_k}: "
            f"{recall:.3f} ({recall * 100:.1f}%)"
        )

    return results
