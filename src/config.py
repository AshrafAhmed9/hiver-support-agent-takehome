"""Single place for model IDs, thresholds, and the cost model.

Three different model families by construction: the generator (OpenAI-oss),
the judge (Allam, SDAIA/IBM), and the golden-set pre-annotator (Qwen) never
share a lineage, so no single model family's bias can silently inflate every
number in the report. All three happen to be served by Groq — that's an
infra choice, not a family choice, and doesn't reintroduce self-preference
risk between generator and judge.

Note on how the judge model was chosen: the original design used Gemini.
Gemini's free tier caps at 20 requests/day per model project-wide —
discovered when the eval run needed ~1,000+ judge calls. Moved to Groq's
`groq/compound` next, which is a genuinely distinct family but is a slow,
tool-using agentic model capped at 30 requests/minute — impractical for a
batch eval. Landed on `allam-2-7b`: distinct family, plain fast chat
completion, no RPM wall hit in testing. See DECISIONS.md.
"""

from __future__ import annotations

BRAND = "SpotifyCares"

GENERATOR_MODEL = "openai/gpt-oss-120b"      # Groq, OpenAI-oss family — drafts replies
JUDGE_MODEL = "allam-2-7b"                    # Groq, Allam family (SDAIA/IBM) — scores reply quality
PRE_ANNOTATOR_MODEL = "qwen/qwen3.8-27b"      # Groq, Qwen family — golden-set label drafts

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
