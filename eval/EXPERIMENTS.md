# Retrieval experiments

What we tried for ranking, with the numbers, so nobody repeats the same experiments.

**Setup.** Calculus Volume 1 (55 modules, 1,607 chunks) loaded with the parser and analyzer in this branch.
`hit@k` is the number of questions where an expected module appears in the top k results; `MRR` is the mean
reciprocal rank (1.0 = right module always first). Questions: `eval/questions.json` (27, used for tuning) and
`eval/holdout.json` (9, kept apart). Expected modules were chosen from the book's outline (`--outline`), not
from search results, and are judgment calls.

Reproduce:

```
python -m search.evaluate --save eval/baseline.json
python -m search.evaluate --fields "text^3,section" --baseline eval/baseline.json
python -m search.evaluate --demote 0.5 --baseline eval/baseline.json
python -m search.evaluate --questions eval/holdout.json
```

## Current set (27 tuning questions)

| Search | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| text only (default) | 20 | 24 | 24 | 0.80 |
| text only, demote Exercises/Glossary x0.5 | 19 | 23 | 24 | 0.77 |
| text only, demote Exercises/Glossary x0.2 | 19 | 23 | 24 | 0.77 |
| `text^3,section` | 18 | 25 | 25 | 0.78 |

Holdout (9 questions), text only: hit@1 4, hit@3 5, hit@5 6, MRR 0.51. Variants were not run on the holdout
because none beat the default.

## Earlier set (13 questions)

| Search | hit@1 | hit@3 | hit@5 | MRR |
| --- | --- | --- | --- | --- |
| text only, before the math analyzer | 10 | 12 | 12 | 0.83 |
| text only, with the math analyzer (`x^2` indexed as "x squared") | 11 | 13 | 13 | 0.91 |
| `text,section` | 11 | 12 | 12 | 0.88 |
| `text,section^2` | 10 | 10 | 11 | 0.79 |
| `text,section^3,title^2,chapter` | 10 | 10 | 10 | 0.77 |
| `text,section^2`, best_fields | 9 | 9 | 9 | 0.69 |

## What we learned

- The math analyzer was the one clear win: a plain `x^2` is now found by "x squared".
- Searching `section`, `title` or `chapter` too does not beat plain text matching. Those fields are short, so
  generic query words ("derivative", "function") pull in the wrong sections.
- Demoting `Exercises`/`Glossary` chunks fixed "limit as x goes to infinity" but lost "water level rising as a
  tank fills", whose only match in the right module was an exercise chunk.
- The remaining misses are semantic ("find dy/dx when x and y are mixed together" should find implicit
  differentiation; "how are derivatives and integrals related" should find the Fundamental Theorem). Keyword
  tweaks will not fix these; the next step is embeddings, combined with keyword search.
- Caveats: small question sets, and one question moving is 1/27. Treat differences of one or two questions as noise.