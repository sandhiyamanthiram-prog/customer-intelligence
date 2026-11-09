import pandas as pd
import re

from guardrails import (
    redact,
    EMAIL_PATTERN,
    PHONE_PATTERN,
    ORDER_PATTERN,
    is_empty,
    is_long
)

tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

email_total = 0
phone_total = 0
order_total = 0

empty_count = 0
long_count = 0

for text in tickets["text"]:

    if is_empty(text):
        empty_count += 1

    if is_long(text):
        long_count += 1

    redacted_text, emails, phones, orders = redact(text)

    email_total += emails
    phone_total += phones
    order_total += orders

    assert len(
        re.findall(
            EMAIL_PATTERN,
            redacted_text
        )
    ) == 0

    assert len(
        re.findall(
            PHONE_PATTERN,
            redacted_text
        )
    ) == 0

    assert len(
        re.findall(
            ORDER_PATTERN,
            redacted_text
        )
    ) == 0

print("Empty Tickets:", empty_count)
print("Long Tickets:", long_count)

print("Emails:", email_total)
print("Phones:", phone_total)
print("Orders:", order_total)

print(
    "Total:",
    email_total
    + phone_total
    + order_total
)

print("\nAll Redaction Tests Passed")