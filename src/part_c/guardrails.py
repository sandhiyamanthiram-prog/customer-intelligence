import re

MAX_INPUT_TOKENS = 400

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def est_tokens(text):
    return len(str(text).split())


def is_empty(text):
    if text is None:
        return True
    return len(str(text).strip()) == 0


def is_long(text):
    return est_tokens(text) > MAX_INPUT_TOKENS


def redact(text):

    email_count = len(
        re.findall(EMAIL_PATTERN, str(text))
    )

    phone_count = len(
        re.findall(PHONE_PATTERN, str(text))
    )

    order_count = len(
        re.findall(ORDER_PATTERN, str(text))
    )

    text = re.sub(
        EMAIL_PATTERN,
        "[EMAIL]",
        str(text)
    )

    text = re.sub(
        PHONE_PATTERN,
        "[PHONE]",
        text
    )

    text = re.sub(
        ORDER_PATTERN,
        "[ORDER_ID]",
        text
    )

    return (
        text,
        email_count,
        phone_count,
        order_count
    )