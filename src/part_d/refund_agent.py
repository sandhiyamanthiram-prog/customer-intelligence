import os, sys, json, sqlite3, csv, re, time, uuid
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)                    # must come BEFORE import mock_llm
sys.path.insert(0, os.path.dirname(__file__))

import mock_llm
from rag_assistant import retrieve

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
STATE_DIR = os.path.join(PROJECT_ROOT, "state")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

# Project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from rag_assistant import retrieve


# 1. GET ORDER
def get_order(order_id):

    db_path = os.path.join(
        PROJECT_ROOT,
        "data/instamart.db"
    )

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            order_id,
            status,
            order_amount,
            delivery_minutes
        FROM orders
        WHERE order_id = ?
        """,
        (order_id,)
    )

    row = cur.fetchone()
    conn.close()

    if row is None:
        return {
            "error": "order not found"
        }

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": row[2],
        "delivery_minutes": row[3]
    }


# 2. SEARCH POLICY
def search_policy(query):

    return {
        "results": retrieve(
            query,
            k=3
        )
    }


# 3. CALCULATE REFUND
def calculate_refund(
    order_amount,
    delivery_minutes
):

     if delivery_minutes > 45:
        refund = round(float(order_amount), 2)
     else:
        refund = 0.0

     return {
        "refund_inr": refund
    }


# 4. REFUND LEDGER
LEDGER = os.path.join(
    PROJECT_ROOT,
    "outputs/refund_ledger.jsonl"
)


def write_ledger(tool_name, payload):

    os.makedirs(
        os.path.dirname(LEDGER),
        exist_ok=True
    )

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        **payload
    }

    with open(
        LEDGER,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(record) + "\n"
        )


# 5. ISSUE REFUND
def issue_refund(order_id, refund_inr):

    write_ledger(
        "issue_refund",
        {
            "order_id": order_id,
            "refund_inr": refund_inr
        }
    )

    return {
        "refund_inr": refund_inr
    }


# 6. ESCALATE TO HUMAN
def escalate_to_human(order_id, refund_inr):

    write_ledger(
        "escalate_to_human",
        {
            "order_id": order_id,
            "refund_inr": refund_inr
        }
    )

    return {
        "refund_inr": refund_inr
    }


# 7. TOOL SPECIFICATIONS
TOOLS_SPEC = [

    {
        "name": "get_order",
        "description": "Retrieve order details using order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string"
                }
            },
            "required": ["order_id"]
        }
    },

    {
        "name": "search_policy",
        "description": "Search policy documents using a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                }
            },
            "required": ["query"]
        }
    },

    {
        "name": "calculate_refund",
        "description": "Calculate refund based on order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number"
                },
                "delivery_minutes": {
                    "type": "number"
                }
            },
            "required": [
                "order_amount",
                "delivery_minutes"
            ]
        }
    },

    {
        "name": "issue_refund",
        "description": "Record an approved automatic refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string"
                },
                "refund_inr": {
                    "type": "number"
                }
            },
            "required": [
                "order_id",
                "refund_inr"
            ]
        }
    },

    {
        "name": "escalate_to_human",
        "description": "Escalate a refund above INR 1000 to a human support lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string"
                },
                "refund_inr": {
                    "type": "number"
                }
            },
            "required": [
                "order_id",
                "refund_inr"
            ]
        }
    }
]


# 8. TOOL MAP
TOOL_MAP = {

    "get_order": get_order,

    "search_policy": search_policy,

    "calculate_refund": calculate_refund,

    "issue_refund": issue_refund,

    "escalate_to_human": escalate_to_human
}

def _save_checkpoint(st):
    with open(f"state/{st['run_id']}.json", "w", encoding="utf-8") as f:
        json.dump({"run_id": st["run_id"], "scenario_id": st["scenario_id"],
                   "messages": st["messages"], "tokens_used": st["tokens_used"],
                   "step": len(st["executed"]), "executed": st["executed"],
                   "llm_calls": st["llm_calls"], "steps": st["steps"]}, f, indent=2)

def _log_run(st, status, final_answer):
    with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"scenario_id": st["scenario_id"], "status": status,
                            "final_answer": final_answer, "llm_calls": st["llm_calls"],
                            "tokens_used": st["tokens_used"], "steps": st["steps"]}) + "\n")

def _finish(st, text):
    status = text.split(":")[0].strip()
    _log_run(st, status, text)
    return {"status": status, "final_answer": text, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}

def _fallback(st, status):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    text = (f"{status}: stopped after {st['llm_calls']} model calls. "
            f"Steps completed: {done}. Please hand this case to a human agent.")
    _log_run(st, status, text)
    return {"status": status, "final_answer": text, "steps": st["steps"],
            "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"]}


# 9. TEST
if __name__ == "__main__":

    print("\nTesting policy search:")

    print(
        search_policy("refund policy")
    )

    print("\nTesting refund calculation:")

    print(
        calculate_refund(
            500,
            60
        )
    )

    print("\nAvailable tools:")

    print(
        list(TOOL_MAP.keys())
    )