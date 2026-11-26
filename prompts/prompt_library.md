# FraudLens — Bibliothèque de prompts versionnée

**Livrable examen · Advanced Deep Learning · Esprit 2026-2027**

---

## Conventions

- **Version active** : v2 (contrôlée par `config/settings.yaml → prompts.active_version`)
- **Fichiers source** : `prompts/v1/*.txt` (before) et `prompts/v2/*.txt` (after)
- Chaque prompt est chargé dynamiquement en Python — aucun prompt n'est hardcodé.
- L'historique complet des échecs et corrections est dans `prompts/CHANGELOG.md`.

---

## Prompt 1 — extract.txt (Vision Agent)

**Rôle dans le système** : Premier nœud de la chaîne. Gemini multimodal (`gemini-3.5-flash-lite`) extrait le texte brut de l'image uploadée par l'utilisateur. Le texte extrait devient la requête d'entrée des agents suivants.

**Technique utilisée (v2)** : Role prompting + guardrails (liste de prohibitions explicites) + sentinel NO_TEXT_FOUND.

---

### Version 1 (BEFORE)

```
Extract the text from this image.
```

**Cas d'échec observés** :
- F-EXT-01 : Le modèle ajoutait une narration ("This image shows a message saying...") au lieu de retourner du texte brut → le JSON parser en aval crashait.
- F-EXT-02 : Sur une image sans texte, le modèle retournait "I don't see any text" → cette chaîne était envoyée à Qdrant comme requête, produisant des résultats absurdes.
- F-EXT-03 : Le modèle ajoutait son interprétation ("this appears to be a phishing attempt") → contamination de l'input de l'Analysis Agent avec une conclusion déjà formée.

---

### Version 2 (AFTER — active)

```
You are an OCR expert. Your ONLY task is to extract ALL visible text from the provided image,
character by character, exactly as it appears on screen or paper.

STRICT RULES:
- Return ONLY the extracted text. No introduction, no commentary, no analysis.
- Preserve original line breaks and paragraph breaks.
- If a word is partially visible or blurry, write the visible part and append [?].
- If the image contains absolutely no text at all, return exactly: NO_TEXT_FOUND
- Do NOT describe the image. Do NOT say "The image shows..." or "This message says...".
- Do NOT interpret, translate, or summarize the content.
- Do NOT add punctuation that is not in the original.
```

**Corrections apportées** :
- Role prompting : "You are an OCR expert" → réduit les narrations.
- Liste de prohibitions explicites → adresse F-EXT-01 et F-EXT-03.
- Sentinel `NO_TEXT_FOUND` géré dans `backend/agents/extract.py` → adresse F-EXT-02.

---

## Prompt 2 — analyze.txt (Analysis Agent)

**Rôle dans le système** : Nœud central. Gemini (`gemini-3.5-flash-lite`) reçoit le texte extrait + les passages RAG et produit un verdict JSON structuré (verdict, score de confiance, type de fraude, raisons, actions, sources citées).

**Technique utilisée (v2)** : Chain-of-thought (5 étapes) + few-shot (3 exemples) + structured JSON output + guardrails (contraintes dures).

---

### Version 1 (BEFORE)

```
Is this message a scam? Explain why.

Message:
{message}

Context from database:
{context}

Answer with: verdict (SCAM/LEGITIMATE), reasons, and what to do.
```

**Cas d'échec observés** :
- F-ANA-01 : `confidence_score: 1.0` retourné sur presque chaque SCAM → rejet systématique par le Validator → boucle de retry.
- F-ANA-02 : `cited_sources` vide — le modèle ignorait le contexte RAG.
- F-ANA-03 : Output JSON encadré de balises markdown (\`\`\`json...\`\`\`) → `json.loads()` échouait.
- F-ANA-04 : Seulement 2 verdicts possibles (SCAM/LEGITIMATE) → cas ambigus forcés en SCAM → taux de faux positifs élevé.
- F-ANA-05 : "vous pouvez récupérer votre argent" dans les actions → promesse de remboursement dangereuse.

---

### Version 2 (AFTER — active)

*(contenu complet dans `prompts/v2/analyze.txt`)*

Techniques combinées :
1. **Chain-of-thought** : 5 étapes de raisonnement avant l'output (urgence → sender → paiement → comparaison RAG → verdict).
2. **Few-shot** : 3 exemples annotés (SCAM, LEGITIMATE, SUSPICIOUS) avec `cited_sources` renseignées → apprend le format attendu.
3. **Structured JSON output** : format explicite avec types attendus pour chaque champ.
4. **Guardrails** (contraintes dures) :
   - `confidence_score` doit être < 1.0 (jamais 1.0) → adresse F-ANA-01.
   - `cited_sources` doit contenir ≥ 1 extrait → adresse F-ANA-02.
   - Pas de balises markdown → adresse F-ANA-03 (+ gestion défensive dans le code).
   - Ajout du verdict SUSPICIOUS → adresse F-ANA-04.
   - Interdiction de promettre un remboursement → adresse F-ANA-05.

**Paramètre LLM** : `temperature: 0.1` (output quasi-déterministe pour la classification).

---

## Prompt 3 — validate.txt (Validator Agent)

**Rôle dans le système** : Dernier nœud. Vérifie que l'analyse respecte 5 guardrails. Si échec, déclenche un retry unique dans `run_analyze(is_retry=True)`. Retourne toujours un résultat (original valide ou régénéré).

**Technique utilisée (v2)** : Role prompting + checklist numérotée avec conditions PASS/FAIL explicites + structured JSON output.

---

### Version 1 (BEFORE)

```
Is this analysis correct? Review it and tell me if anything is wrong.
Analysis: {analysis}
```

**Cas d'échec observés** :
- F-VAL-01 : Le modèle répondait de façon conversationnelle ("Yes, the analysis looks good overall...") → `json.JSONDecodeError`.
- F-VAL-02 : Aucun critère fourni → le modèle inventait ses propres checks (grammaire, ton).
- F-VAL-03 : Faux rejets aléatoires → retries inutiles.

---

### Version 2 (AFTER — active)

*(contenu complet dans `prompts/v2/validate.txt`)*

5 checks implémentés :
1. `CONFIDENCE_BELOW_MAX` : `confidence_score < 1.0`
2. `SOURCES_CITED` : `cited_sources` non vide et correspondant aux passages
3. `NO_REFUND_PROMISE` : absence de mots-clés de remboursement dans les actions
4. `REASONS_SUFFICIENT` : au moins 2 raisons distinctes
5. `ACTIONS_PRESENT` : au moins 1 action recommandée

**Note architecturale (validation hybride)** : Les checks 1, 3, 4 et 5 sont vérifiables mécaniquement : ils sont décidés par le code (`_run_code_checks()` dans `backend/agents/validate.py`), ce qui évite les faux rejets (F-VAL-03). Le check 2 a une partie sémantique que le code ne peut pas juger (« la citation correspond-elle vraiment à un passage RAG ? ») : c'est le seul point où la réponse du LLM (prompt v2, `temperature: 0`, sortie JSON forcée via `response_mime_type="application/json"`) peut faire échouer la validation. Si l'appel LLM échoue (quota, saturation), le Validator se rabat sur les checks en code — le pipeline ne casse jamais à cette étape.

---

## Résumé des techniques par prompt

| Prompt | Technique principale | Techniques secondaires |
|--------|---------------------|----------------------|
| extract.txt v2 | Role prompting | Guardrails (prohibitions), sentinel |
| analyze.txt v2 | Chain-of-thought | Few-shot (3 ex.), structured output, guardrails |
| validate.txt v2 | Structured output | Checklist PASS/FAIL, role prompting |
