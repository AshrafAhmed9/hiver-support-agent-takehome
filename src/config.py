"""Single place for model IDs, thresholds, and the cost model.

Three different model families by construction: the generator, the judge, and
the golden-set pre-annotator never share a family, so no single provider's
bias can silently inflate every number in the report.
"""

from __future__ import annotations

BRAND = "SpotifyCares"

GENERATOR_MODEL = "openai/gpt-oss-120b"      # Groq — drafts replies
JUDGE_MODEL = "gemini-2.5-pro"                # Gemini — scores reply quality
PRE_ANNOTATOR_MODEL = "qwen/qwen3.8-27b"      # Groq (Qwen family) — golden-set label drafts

# Cost model for deriving the escalation threshold (§9.3 / §6 of the plan).
# Units are arbitrary and relative to each other, not currency.
COST_BAD_AUTO_REPLY = 8.0   # customer harm + rework when the machine gets it wrong
COST_HUMAN_TOUCH = 1.0      # marginal cost of a human agent handling one ticket

# Sensitivity sweep for the report: how the headline coverage number moves
# as the assumed cost ratio changes.
COST_RATIO_SWEEP = [2.0, 4.0, 8.0, 12.0, 20.0]

TARGET_SAFE_AUTO_REPLY_RATE = 0.95

RANDOM_SEED = 20260909

INTENTS = [
    "playback_or_app_bug",
    "login_or_account_access",
    "billing_or_subscription_charge",
    "content_availability_or_licensing",
    "feature_request_or_product_feedback",
    "device_or_platform_compatibility",
    "account_data_or_privacy",
    "how_to_or_usage_question",
    "other",
    "not_a_support_request",
]
