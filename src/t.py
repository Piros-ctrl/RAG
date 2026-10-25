import fire


def hello(name):
    return f'Hello {name}!'


def index(k):
    return f"indixing Completed {k}"


if __name__ == '__main__':
    fire.Fire()



'import json

with open("data/search_results/search_results.json") as f:
    student = json.load(f)["search_results"]

with open("data/datasets/public/AnsweredQuestions/dataset_docs_public.json") as f:
    dataset = json.load(f)["rag_questions"]

data = [d for d in dataset["question"]]
stud = [st fot st in dataset["question"]]

print("\nFirst student ID:", stud[0])
print("First dataset ID:", data[0])'
