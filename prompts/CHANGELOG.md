# Prompt Changelog — FraudLens

This file tracks every iteration of every prompt: what changed, why, and what failure triggered the change.

---

## extract.txt

### v1 (initial)
```
Extract the text from this image.
```

### Failures observed with v1
- **F-EXT-01**: The model prepended narrative ("This image shows a text message saying...") instead of returning raw text. The downstream JSON parser received garbage and crashed.
- **F-EXT-02**: On a blank/logo-only image, the model returned "I don't see any text" — the string "I don't see any text" was then sent to Qdrant as a query, producing irrelevant retrieval results.
- **F-EXT-03**: The model added interpretive notes ("this appears to be a phishing attempt") which poisoned the Analysis Agent's input with a pre-formed conclusion.

### v2 (current) — fixes applied
- Added **role prompting**: "You are an OCR expert. Your ONLY task is..."
- Added **explicit prohibitions**: "Do NOT describe the image. Do NOT say 'The image shows...'."
- Added **NO_TEXT_FOUND sentinel**: handled in `backend/agents/extract.py` to short-circuit the pipeline cleanly.
- Added **[?] notation** for partially visible words.

---

## analyze.txt

### v1 (initial)
```
Is this message a scam? Explain why.
Message: {message}
Context: {context}
Answer with: verdict (SCAM/LEGITIMATE), reasons, and what to do.
```

### Failures observed with v1
- **F-ANA-01**: The model returned `confidence_score: 1.0` on almost every SCAM verdict. The Validator rejected these, causing infinite retry loops until the retry cap was hit.
- **F-ANA-02**: `cited_sources` was always empty — the model ignored the context. The Validator rejected all outputs as "sources not cited."
- **F-ANA-03**: The model wrapped the JSON in markdown code fences (` ```json ... ``` `), breaking `json.loads()`.
- **F-ANA-04**: v1 only used two verdicts (SCAM/LEGITIMATE). Edge cases like ambiguous messages were forced into SCAM, inflating false positives.
- **F-ANA-05**: The model occasionally included "vous pouvez récupérer votre argent" in recommended_actions — a dangerous refund promise the Validator was designed to catch.

### v2 (current) — fixes applied
- Added **chain-of-thought**: 5-step reasoning process before producing output.
- Added **three few-shot examples** (SCAM, LEGITIMATE, SUSPICIOUS) — directly addresses F-ANA-02 by showing cited_sources in examples.
- Added **SUSPICIOUS verdict** — addresses F-ANA-04, reduces false positive rate.
- Added explicit **hard constraint**: "confidence_score MUST be strictly less than 1.0" — addresses F-ANA-01.
- Added explicit **hard constraint**: "cited_sources MUST contain at least one excerpt" — addresses F-ANA-02.
- Added explicit **hard constraint**: "Do NOT suggest financial recovery" — addresses F-ANA-05.
- Added **"no markdown fences"** instruction — addresses F-ANA-03 (also handled defensively in code).

---

## validate.txt

### v1 (initial)
```
Is this analysis correct? Review it and tell me if anything is wrong.
Analysis: {analysis}
```

### Failures observed with v1
- **F-VAL-01**: The model responded conversationally ("Yes, the analysis looks good overall, though you might want to..."). The code tried to parse this as JSON and raised `json.JSONDecodeError`.
- **F-VAL-02**: v1 gave no criteria — the model invented its own checks (grammar, tone) and ignored the actual guardrails.
- **F-VAL-03**: When all checks should have passed, the model sometimes returned "I see a few minor issues..." causing false rejections and unnecessary retries.

### v2 (current) — fixes applied
- Added **explicit numbered checklist** with PASS/FAIL conditions — addresses F-VAL-02.
- Added **strict JSON-only output** instruction — addresses F-VAL-01.
- Added **concrete examples** of what counts as a refund promise (specific word list) — removes ambiguity.
- Added **knowledge base passages** as context so the model can actually verify citations — addresses F-VAL-02.
- Defensive JSON extraction also implemented in `backend/agents/validate.py` as a code-level safety net.
