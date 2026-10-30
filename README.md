# MLExperimentalCodex

Small experiments and explorations. Each folder is self-contained, with its own setup.

| Folder | What it is | Date | Status |
| --- | --- | --- | --- |
| [`agents-playground/grounding`](agents-playground/grounding) | Behavioural tests for a tool-using LLM agent | July 2026 | draft |
| [`experiments/ionosphere`](experiments/ionosphere) | Class-weighting experiments for the Class Imbalance article | 2023 | archive |
| [`experiments/HomeCreditDefaultRisk`](experiments/HomeCreditDefaultRisk) | Data-understanding notebook for a Kaggle credit-default competition | 2025 | first step only |

## agents-playground/grounding

These tests check whether a tool-using agent trusts its tools over its own memory, and whether it admits it doesn't know when the tools return nothing. The agent runs locally through [Ollama](https://ollama.com) with `qwen3-vl:30b` at temperature 0.

`test_grounding.py` has three checks:

- **Grounding ("lie test").** The search tool returns a deliberately false fact: that Monet painted *The Starry Night*. The check passes if the agent repeats the tool's answer instead of the one it knows from training.
- **Abstention.** The search tool returns no results. The check passes if the agent does not answer from memory.
- **Tool routing.** Given four queries, the check passes if the agent's first tool call goes to the right tool: `web_search` for facts, or `find_in_artic` (the Art Institute of Chicago API) for viewable artworks.

The grounding and abstention checks each run 3 times and pass when at least 2 of the 3 runs pass. That makes this a smoke test, not a measurement. Three runs can't support a pass rate.

**Known issue.** `agent.py` only registers `web_search`, so the two routing cases that expect `find_in_artic` can't pass yet.

**To run:**

```bash
pip install ollama ddgs requests pytest
ollama pull qwen3-vl:30b
cd agents-playground/grounding
pytest -s test_grounding.py
```

## experiments/ionosphere

Logistic regression on the [UCI Ionosphere dataset](https://archive.ics.uci.edu/dataset/52/ionosphere), comparing class weights, training time, precision and test accuracy. It accompanies the Class Imbalance article ([blog](https://www.nonneutralzero.com/blog/analytics-data-6/class-imbalance-18), [articles](https://non-neutralzero.github.io/articles/article-class-imbalance/)).

## experiments/HomeCreditDefaultRisk

Data understanding for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition. Only the first step, exploring the data, is here. The data isn't included: download it from Kaggle. The folder's README has the Docker setup.
