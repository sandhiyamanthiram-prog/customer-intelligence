import os, sys, json, csv, time, uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import mock_llm
from refund_agent import TOOLS_SPEC, TOOL_MAP

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
STATE_DIR = os.path.join(PROJECT_ROOT, "state")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

AGENT_PROMPT = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
                "calculate the refund, then issue or escalate it. Finish with STATUS: details.")


class SimulatedCrash(Exception):
    def __init__(self, run_id):
        super().__init__("simulated crash")
        self.run_id = run_id


def _save_checkpoint(st):
    with open(os.path.join(STATE_DIR, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": st["run_id"], "scenario_id": st["scenario_id"],
                   "messages": st["messages"], "tokens_used": st["tokens_used"],
                   "step": len(st["executed"]), "executed": st["executed"],
                   "llm_calls": st["llm_calls"], "steps": st["steps"]}, f, indent=2)


def _log_run(st, status, final_answer):
    with open(os.path.join(OUT_DIR, "agent_runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": st["scenario_id"], "status": status,
                            "final_answer": final_answer, "llm_calls": st["llm_calls"],
                            "tokens_used": st["tokens_used"], "steps": st["steps"]}) + "\n")


def _result(st, status, text):
    _log_run(st, status, text)
    return {"status": status, "final_answer": text, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}


def _finish(st, text):
    return _result(st, text.split(":")[0].strip(), text)


def _fallback(st, status):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    text = (f"{status}: stopped after {st['llm_calls']} model calls. "
            f"Steps completed: {done}. Please hand this case to a human agent.")
    return _result(st, status, text)


def _loop(st, max_it, tok_lim, time_lim, crash_after=None):
    t0 = time.monotonic()
    while True:
        if time.monotonic() - t0 >= time_lim:
            return _fallback(st, "FALLBACK_TIMEOUT")
        if st["llm_calls"] >= max_it:
            return _fallback(st, "FALLBACK_MAX_ITERATIONS")

        resp = mock_llm.chat(st["messages"], TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > tok_lim:
            return _fallback(st, "FALLBACK_TOKEN_LIMIT")

        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _finish(st, msg.content)

        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name,
                                               "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id,
                               "name": tc.name, "content": json.dumps(result)})
        st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
        st["executed"].append(tc.name)
        _save_checkpoint(st)
        if crash_after and len(st["executed"]) == crash_after:
            raise SimulatedCrash(st["run_id"])


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, crash_after_tools=None):
    st = {"run_id": f"{scenario_id}-{uuid.uuid4().hex[:6]}", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": AGENT_PROMPT},
                       {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "steps": [], "executed": []}
    return _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        st = json.load(f)
    return _loop(st, max_iterations, token_limit, time_limit_seconds)


def _load_scenarios():
    path = os.path.join(PROJECT_ROOT, "data", "raw", "agent_scenarios.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _refund_amount(res):
    for s in reversed(res["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return None


def run_all():
    print("scenario | status | refund | calls | tokens | expected_outcome | expected_refund")
    for r in _load_scenarios():
        mock_llm.reset()
        res = run_agent(r["message"], r["scenario_id"])
        amt = _refund_amount(res)
        amt_txt = f"{amt:.2f}" if amt is not None else "-"
        print(r["scenario_id"], res["status"], amt_txt, res["llm_calls"],
              res["tokens_used"], r["expected_outcome"], r["expected_refund_inr"])


def run_demos():
    s1 = next(r for r in _load_scenarios() if r["scenario_id"] == "S1")
    print("\nToken limit demo:", run_agent(s1["message"], "S1", token_limit=200)["status"])
    print("Timeout demo:", run_agent(s1["message"], "S1", time_limit_seconds=0)["status"])
    try:
        run_agent(s1["message"], "S1", crash_after_tools=2)
    except SimulatedCrash as e:
        print("Crashed, resuming run", e.run_id)
        res = resume_agent(e.run_id)
        with open(os.path.join(STATE_DIR, f"{e.run_id}.json"), encoding="utf-8") as f:
            executed = json.load(f)["executed"]
        print("Tools executed across crash + resume:", executed)
        print("Final:", res["status"], _refund_amount(res))


if __name__ == "__main__":
    run_all()
    run_demos()