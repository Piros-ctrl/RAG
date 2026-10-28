PYTHON   = uv run python
MODULE   = src
ARGS     = index

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m $(MODULE)

debug:
	$(PYTHON) -m pdb -m $(MODULE) $(ARGS)

clean:
	find . -path ./.venv -prune -o -type d \( -name __pycache__ -o -name .mypy_cache -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	find . -path ./.venv -prune -o -type f -name "*.pyc" -exec rm -f {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

index:
	$(PYTHON) -m $(MODULE) $(ARGS) --max_chunk_size 1500

search:
	$(PYTHON) -m $(MODULE) search "What does the vllm mean" --k 10

search-dataset:
	$(PYTHON) -m $(MODULE) search_dataset \
	--dataset_path data/datasets/public/UnansweredQuestions/dataset_code_public.json \
	--k 10 \
	--save_directory data/search_results

answer:
	$(PYTHON) -m $(MODULE) answer "what's the capital of morocco" --k 10

answer-dataset:
	$(PYTHON) -m $(MODULE) answer_dataset \
  	--student_search_result_path data/search_results/search_results.json \
  	--save_directory data/output \
	--k 10

evaluate:
	$(PYTHON) -m $(MODULE) evaluate \
	--student_search_result_path data/search_results/dataset_code_public.json \
	--dataset_path data/datasets/public/AnsweredQuestions/dataset_code_public.json \
	--k 10
