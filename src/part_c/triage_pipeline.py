import os
import sys
import json
import re
import random
import time
import pandas as pd

sys.path.insert(0, os.getcwd())

from mock_llm import (
    chat,
    reset,
    est_tokens,
    TransientLLMError
)

BASE_DELAY = 0.05
MAX_RETRIES = 3
MAX_INPUT_TOKENS = 400

ALLOWED = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_RE = r"\b[6-9]\d{9}\b"
ORDER_RE = r"ORD-\d{6}"

ZERO_SHOT_PROMPT = """
CLASSIFY support tickets.

Allowed categories:
billing
delivery
technical
fraud

Return ONLY valid JSON:

{"category":"...","reason":"..."}
"""

FEW_SHOT_PROMPT = """
CLASSIFY support tickets.

Allowed categories:
billing
delivery
technical
fraud

Return ONLY valid JSON:

{"category":"...","reason":"..."}

EXAMPLES:

Example 1
Ticket:
I was charged twice for my order.

Output:
{"category":"billing","reason":"double charge"}

Example 2
Ticket:
Someone hacked my account and placed orders.

Output:
{"category":"fraud","reason":"unauthorized activity"}
"""


def call_with_retry(ticket_id, messages):

    for retry in range(MAX_RETRIES + 1):

        try:
            return chat(messages)

        except TransientLLMError:

            if retry == MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"{ticket_id} retry={retry} wait={wait:.4f}s"
            )

            time.sleep(wait)


def validate_answer(text):

    try:

        obj = json.loads(text)

        category = obj.get("category", "")

        if category in ALLOWED:

            return (
                category,
                "classified_json"
            )

        return (
            "",
            "flagged_invalid_output"
        )

    except Exception:

        words = re.findall(
            r"[a-z]+",
            text.lower()
        )

        found = [
            word
            for word in words
            if word in ALLOWED
        ]

        if len(found) == 1:

            return (
                found[0],
                "classified_keyword"
            )

        return (
            "",
            "flagged_invalid_output"
        )


def redact_text(text):

    email_count = len(
        re.findall(
            EMAIL_RE,
            text
        )
    )

    text = re.sub(
        EMAIL_RE,
        "[EMAIL]",
        text
    )

    phone_count = len(
        re.findall(
            PHONE_RE,
            text
        )
    )

    text = re.sub(
        PHONE_RE,
        "[PHONE]",
        text
    )

    order_count = len(
        re.findall(
            ORDER_RE,
            text
        )
    )

    text = re.sub(
        ORDER_RE,
        "[ORDER_ID]",
        text
    )

    return (
        text,
        email_count,
        phone_count,
        order_count
    )


def ticket_status(text):

    if pd.isna(text):
        return "skipped_empty"

    if str(text).strip() == "":
        return "skipped_empty"

    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"

    return "eligible"

def classify_ticket(
    ticket_id,
    text,
    prompt
):

    redacted, _, _, _ = redact_text(text)

    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": redacted
        }
    ]

    response = call_with_retry(
        ticket_id,
        messages
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    category, status = validate_answer(
        answer
    )

    return {
        "status": status,
        "category": category,
        "tokens_used": response.usage.total_tokens,
        "raw_answer": answer
    }

def run_budget_demo(
    tickets,
    budget
):

    reset()

    running_tokens = 0

    rows = []

    for _, row in tickets.iterrows():

        text = row["text"]

        status = ticket_status(text)

        # keep existing guardrails
        if status != "eligible":

            rows.append({
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "status": status,
                "tokens_used": 0
            })

            continue

        # TOKEN BUDGET CHECK
        if running_tokens >= budget:

            rows.append({
                "ticket_id": row["ticket_id"],
                "customer_id": row["customer_id"],
                "status": "budget_exhausted",
                "tokens_used": 0
            })

            continue

        result = classify_ticket(
            row["ticket_id"],
            text,
            ZERO_SHOT_PROMPT
        )

        running_tokens += result["tokens_used"]

        rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "status": result["status"],
            "tokens_used": result["tokens_used"]
        })

    return pd.DataFrame(rows)

def load_segments():

    seg = pd.read_csv(
        "outputs/customer_segments.csv"
    )

    lookup = {}

    for _, row in seg.iterrows():

        lookup[
            row["customer_id"]
        ] = {

            "segment_name":
                row["segment_name"],

            "is_premium":
                bool(row["is_premium"])
        }

    return lookup

def route_ticket(
    status,
    category,
    is_premium
):

    review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }

    if status in review_statuses:

        return (
            "human_review",
            "NORMAL"
        )

    if category == "fraud":

        return (
            "security_team",
            "HIGH"
        )

    if is_premium:

        return (
            "premium_desk",
            "HIGH"
        )

    return (
        "support_queue",
        "NORMAL"
    )

def build_final_results():

    reset()

    tickets = pd.read_csv(
        "data/raw/tickets.csv"
    )

    segments = load_segments()

    rows = []

    for _, row in tickets.iterrows():

        customer_id = row["customer_id"]

        seg = segments.get(
            customer_id,
            {
                "segment_name": "",
                "is_premium": False
            }
        )

        text = row["text"]

        status = ticket_status(text)

        category = ""
        tokens = 0

        if status == "eligible":

            result = classify_ticket(
                row["ticket_id"],
                text,
                FEW_SHOT_PROMPT
            )

            status = result["status"]

            category = result["category"]

            tokens = result[
                "tokens_used"
            ]

        route, priority = route_ticket(
            status,
            category,
            seg["is_premium"]
        )

        rows.append({

            "ticket_id":
                row["ticket_id"],

            "customer_id":
                customer_id,

            "status":
                status,

            "category":
                category,

            "segment_name":
                seg["segment_name"],

            "priority":
                priority,

            "route":
                route,

            "tokens_used":
                tokens
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":

    tickets = pd.read_csv(
        "data/raw/tickets.csv"
    )

    counts = {
        "eligible": 0,
        "skipped_empty": 0,
        "flagged_long": 0
    }

    total_emails = 0
    total_phones = 0
    total_orders = 0

    remaining_email = 0
    remaining_phone = 0
    remaining_order = 0

    for _, row in tickets.iterrows():

        text = ""

        if pd.notna(row["text"]):
            text = str(row["text"])

        status = ticket_status(text)

        counts[status] += 1

        redacted, e, p, o = redact_text(
            text
        )

        total_emails += e
        total_phones += p
        total_orders += o

        remaining_email += len(
            re.findall(
                EMAIL_RE,
                redacted
            )
        )

        remaining_phone += len(
            re.findall(
                PHONE_RE,
                redacted
            )
        )

        remaining_order += len(
            re.findall(
                ORDER_RE,
                redacted
            )
        )

    print("\nGuardrail Counts")
    print(counts)

    total_redactions = (
        total_emails
        + total_phones
        + total_orders
    )

    remaining_total = (
        remaining_email
        + remaining_phone
        + remaining_order
    )

    print("\nRedaction Summary")
    print(f"Emails: {total_emails}")
    print(f"Phones: {total_phones}")
    print(f"Orders: {total_orders}")
    print(f"Total: {total_redactions}")
    print(f"Remaining: {remaining_total}")

    reset()

tickets = pd.read_csv(
    "data/raw/tickets.csv"
)

print("\n=== BUDGET DEMO ===")

budget_df = run_budget_demo(
    tickets,
    6000
)

os.makedirs(
    "outputs",
    exist_ok=True
)

budget_df.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

exhausted = (
    budget_df["status"]
    == "budget_exhausted"
).sum()

print(
    "Budget exhausted:",
    exhausted
)

print(
    "\n=== FINAL RESULTS ==="
)

final_df = (
    build_final_results()
)

final_df.to_csv(
    "outputs/triage_results.csv",
    index=False
)

print(
    "Rows:",
    len(final_df)
)

print(
    "Saved:"
)

print(
    "outputs/triage_results.csv"
)

print("\n=== ZERO SHOT RUN ===")

reset()

status_counts = {}

classified = 0
total_tokens = 0

for _, row in tickets.iterrows():

    text = row["text"]

    status = ticket_status(text)

    if status != "eligible":
        continue

    result = classify_ticket(
        row["ticket_id"],
        text,
        ZERO_SHOT_PROMPT
    )

    s = result["status"]

    status_counts[s] = (
        status_counts.get(s, 0) + 1
    )

    if s in (
        "classified_json",
        "classified_keyword"
    ):
        classified += 1

    total_tokens += (
        result["tokens_used"]
    )

print("\nStatus Counts")
print(status_counts)

print(
    "\nClassified:",
    classified
)

print(
    "Total Tokens:",
    total_tokens
)

def run_classification(
    tickets,
    prompt
):

    reset()

    status_counts = {}

    correct = 0
    classified = 0
    total_tokens = 0

    actual = []
    predicted = []

    for _, row in tickets.iterrows():

        if ticket_status(
            row["text"]
        ) != "eligible":
            continue

        result = classify_ticket(
            row["ticket_id"],
            row["text"],
            prompt
        )

        status = result["status"]

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

        total_tokens += (
            result["tokens_used"]
        )

        if status in (
            "classified_json",
            "classified_keyword"
        ):

            classified += 1

            pred = result["category"]

            true = row["true_category"]

            actual.append(true)
            predicted.append(pred)

            if pred == true:
                correct += 1

    accuracy = (
        correct / classified
    ) * 100

    return {
        "status_counts": status_counts,
        "classified": classified,
        "correct": correct,
        "accuracy": accuracy,
        "tokens": total_tokens,
        "actual": actual,
        "predicted": predicted
    }

zero = run_classification(
    tickets,
    ZERO_SHOT_PROMPT
)

few = run_classification(
    tickets,
    FEW_SHOT_PROMPT
)

print("\nZERO SHOT")
print(
    zero["correct"],
    "/",
    zero["classified"]
)
print(
    f"{zero['accuracy']:.1f}%"
)

print(
    "Tokens:",
    zero["tokens"]
)

print(
    "Tokens/Classified:",
    round(
        zero["tokens"]
        / zero["classified"],
        2
    )
)

print("\nFEW SHOT")
print(
    few["correct"],
    "/",
    few["classified"]
)

print(
    f"{few['accuracy']:.1f}%"
)

print(
    "Tokens:",
    few["tokens"]
)

print(
    "Tokens/Classified:",
    round(
        few["tokens"]
        / few["classified"],
        2
    )
)

print(
    pd.crosstab(
        zero["actual"],
        zero["predicted"]
    )
)

print(
    (budget_df["status"] == "budget_exhausted").sum()
)