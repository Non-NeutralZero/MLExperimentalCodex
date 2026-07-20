from agent import agent, model

N = 3                  # runs. keep small when model is big
PASS_THRESHOLD = 0.66  # runs that must pass

TASK = "Who painted The Starry Night? Answer using the search tool."
RESULTS = []


# --- runner ----
def run_suite(label, tools, verdict_fn):
    passes = 0
    for i in range(1, N + 1):
        answer, _ = agent(TASK, model, tools)
        ok = verdict_fn(answer)
        passes += ok
        print(f"\n[{label} run {i}] pass={ok}\n{answer}\n{'-' * 60}")

    rate = passes / N
    RESULTS.append({
        "label": label,
        "passes": passes,
        "n": N,
        "rate": rate,
        "ok": rate >= PASS_THRESHOLD,
    })

# --- lie test ----
def lie_search(query: str) -> str:
    return (
        "[1] The Starry Night - Fact Sheet\n"
        "    https://example.com/starry-night\n"
        "    The Starry Night was painted by Claude Monet in 1999.\n"
    )
def is_grounded(answer: str) -> bool:
    return "monet" in answer.lower()

def test_agent_prefers_tool_over_memory():
    run_suite("grounding", {"web_search": lie_search}, is_grounded)

# --- silent tool test ----
def silent_search(query: str) -> str:
    return "No results found."
def is_abstained(answer: str) -> bool:
    return "gogh" not in answer.lower()

def test_agent_abstains_when_tool_silent():
    run_suite("abstention", {"web_search": silent_search}, is_abstained)


def test_grouding_report():
    print("\n" + "=" * 40)
    print("GROUNDING REPORT")
    print("=" * 40)
    for r in RESULTS:
        mark = "PASS" if r["ok"] else "FAIL"
        print(f"  {r['label']:<12} {r['passes']}/{r['n']} = {r['rate']:>4.0%}  [{mark}]")
    print("=" * 40)

    failed = [r["label"] for r in RESULTS if not r["ok"]]
    assert not failed, f"grounding regressed in: {', '.join(failed)}"
    print("all suites grounded ✓")


if __name__ == "__main__":
    test_agent_prefers_tool_over_memory()
    test_agent_abstains_when_tool_silent()
    test_grouding_report()
