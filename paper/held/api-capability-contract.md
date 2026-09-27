# Held material: the API surface is not a capability contract

Cut from the token paper 2026-09-26 — the pattern is a real finding but the
section read partly as an access complaint. Keep for a paper of its own, most
likely alongside the service-tier and model-selection work, where the
rejections are the subject rather than a boundary.

---

## 6. What the API promises and the models reject

Three capabilities central to further work are accepted by the API and rejected
by the models we could invoke:

| capability | API | Sonnet 4.5 / Haiku 4.5 |
|---|---|---|
| `outputConfig.effort` | accepted parameter | `ValidationException: This model doesn't support the effort field` |
| `serviceTier: flex`, `priority` | enum members | `ValidationException: The provided service tier is not supported for this model` |
| evaluation `taskType` | enum lists `Custom`, `Generation` | both rejected; requires `General`, which is absent from the enum |

Separately, `ListInferenceProfiles` reports Claude Opus 5, Opus 5.5, Sonnet 5 and
Fable 5.1 as `ACTIVE` with account-scoped ARNs, while `Converse` against those
same identifiers returns 403 *"not available for this account."*

The through-line is that the API surface is not a capability contract. Parameter
acceptance, enum membership and catalog presence each promise something that the
model or the account may not honour, and the discrepancy appears only at invoke
time. A benchmark that assumes a documented parameter is active will silently
measure its absence — which is the same failure mode as an ignored cache
checkpoint, and argues for the same remedy: assert on the response.

These rejections bound this paper. We cannot report an `effort` curve, nor
measure the `flex` tier at half price, on the models available to us.

---

