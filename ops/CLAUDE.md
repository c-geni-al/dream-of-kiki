# ops/ — submission, reviewer recruitment, outreach

Process docs + Apple Mail automation for journal submission, formal-reviewer recruitment (DR-2 proof external review), and T-Col outreach. No Python here; AppleScript + markdown only.

## Files

| File | Role |
|---|---|
| `nature-hb-submit-tracker.md` | Nature HB Paper 1 submission state, package checklist, gate dependencies |
| `formal-reviewer-recruitment.md` | DR-2 proof reviewer playbook + EN/FR candidate tables |
| `formal-reviewer-email-template.md` | Canonical outreach template (EN + FR variants) |
| `tcol-outreach-plan.md` | T-Col fMRI lab + reviewer outreach matrix |
| `proposal-template-draft.md` | Reusable proposal skeleton for grant / collab asks |
| `create-mail-drafts.applescript` | Generates `.eml` drafts in Apple Mail from candidate tables |
| `create-paper1-v0p2-mail-drafts.applescript` | Same, for Paper 1 v0.2 second-round outreach |

## Conventions

- **Tracker files are append-only**: dated entries (`## 2026-04-20 — <event>`); never rewrite history. Status corrections add a new entry that supersedes.
- **AppleScript is draft-only**: scripts MUST stop at `make new outgoing message` + `set its visible to true` — never `send`. Auto-send violates reviewer-outreach hygiene.
- **Reviewer identity protection**: real names + emails live in tracker tables for the active recruitment cycle; once review is complete, redact to `reviewer-N` pseudonym before public commit / Zenodo snapshot.
- **`.eml` drafts** referenced from tables (`mail-<slug>.eml`) live in `Business OS/` outside the repo — keep the path + slug pinned in the table; do not commit drafted emails into `ops/`.
- **Bilingual outreach**: EN and FR candidate tracks are separate tables (TCS / category-theory pool vs FEP / cognitive-AI pool). Keep them separate; do not merge.

## Coupling

- DR-2 axiom proof status (in `STATUS.md`) ↔ `formal-reviewer-recruitment.md` recruitment state. Paper submission gate (G3-draft / G6-submit) blocks on `Status: external-review-complete` here.
- `nature-hb-submit-tracker.md` checklist items map 1:1 to Nature HB portal fields; stale checklist = stale submission readiness claim.
- `tcol-outreach-plan.md` carries both fMRI lab outreach (Paper 2 / G1 dependency) and reviewer outreach (Paper 1 / G3); the two tracks share a file but not a status.

## AppleScript hygiene

- Run scripts from Apple Mail context (`osascript create-mail-drafts.applescript`); they read candidate tables in this dir and emit `.eml` drafts to a Drafts mailbox outside the repo.
- Scripts are read-mostly: input = markdown tables, output = drafts. No filesystem writes inside the repo. If a script needs to persist state, route through a tracker file via append-only entry.

## Anti-patterns

- Editing past tracker entries instead of appending a new one (loses audit trail; reviewers may have already cited the old state).
- Real reviewer names in plaintext after recruitment closes — pseudonymize before any public-facing commit.
- AppleScript that calls `send` instead of leaving the message in Drafts — outreach must be human-reviewed before transmission.
- Cross-citing a Paper 2 (ablation) reviewer in the Paper 1 (formal) recruitment table — venues + scopes differ.
