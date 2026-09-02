# Banking77 GEPA proposer brief

Optimize only the classification system prompt for a frozen GPT-OSS-20B model.
The model sees one customer utterance and the complete canonical label list. It must
return exactly one label.

Use failures to infer reusable decision boundaries, especially operation type and
state: card payment vs cash withdrawal vs transfer vs direct debit vs top-up, and
pending vs failed vs declined vs reverted vs unrecognized. Distinguish obtaining or
activating a product from troubleshooting an existing one. Preserve exact-label-only
output requirements.

Propose compact, operational guidance that generalizes. Do not memorize utterances,
include gold answers for individual examples, change the label vocabulary, or add
few-shot examples copied from reflection data.
