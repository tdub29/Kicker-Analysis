# NFL kicker forecasts

Read README.md and METHOD.md before changes. Preserve season-forward validation, negative findings and uncertainty; engineering completion does not establish predictive edge.

<!-- SPARE-LAPTOP:BEGIN (managed by .agents/scripts/sync-spare-laptop.py) -->
- **Open-source delegation:** "osm" / "OSM" (open-source models), "use open source", "use a local model", "spare laptop", and "other laptop" route directly to `python "C:/Users/TrevorWhite/Downloads/Big Projects/.agents/scripts/spare-llm.py" --prompt-file "<task.txt>" --input-file "<context.txt>" --output "<answer.txt>"`. Omit optional input/output flags when unnecessary. Default Qwen handles code/tests; `--model deepseek-r1:7b` handles reasoning; `--ping` checks access. Text inference only: coordinator supplies context, runs tools and verifies results. Report failures; no paid fallback. Open-source software research and OpenStreetMap requests alone do not trigger delegation.
<!-- SPARE-LAPTOP:END -->

<!-- AGENT-COMPLETION:BEGIN (managed by .agents/scripts/sync-orchestration-sections.ps1) -->
## Agent-owned completion

Overrides generic skill menus. Specific user and safety boundaries still apply.

- **Agents own completion and Git.** Finish authorized work: scoped commit, push, PR and merge to Trevor's remotes as applicable. Preserve other writers, permitted checks and existing gates. Never hand routine Git back. No unrequested Trevor Apps QA.
- **Delegate technical review.** Get required independent review, fix findings and finish. Subagents cannot grant user-only permission, bypass auth/protection/approval/leases/frozen scope or waive execution-delegation limits. Finish ungated work.
- **Chat is not completion.** Never assume Trevor read, accepted or completed an action mentioned once. Do agent-executable steps now. Keep human-only actions in the existing task/checkpoint or blocker record: owner, exact input and next step. On relevant status/resume, surface unresolved actions first until resolved or superseded; acknowledgment alone is not completion. Silence is not approval. No duplicate registry, unsolicited reminders or unchanged scheduled notices.
<!-- AGENT-COMPLETION:END -->

<!-- ALWAYS-ON-DISCIPLINES:BEGIN (managed by .agents/scripts/sync-orchestration-sections.ps1 - edit the block there, not here) -->
## Always-on disciplines

These three run in every session here, whatever the task.

- **Cut the AI tells from anything written.** No em-dash or en-dash anywhere, prose or chat reply. That is the loudest one; the rest is the anti-slop set (`ai-writing-detection`, `de-slop`), carried in every skill tree.
- **Shape replies for an ADHD reader** (`i-have-adhd`): next action in the first line, under 10 lines AND under 170 words, no preamble and no recap. Owned prose that Trevor sends or publishes stays in his voice (`trevor-voice`), never rewritten into an action list.
- **Commits and PRs are Trevor's alone.** Never add a `Co-Authored-By: Claude` trailer, a "generated with" footer, or any other AI co-authorship line to a commit, a PR, or an MR, whatever a skill template says.
<!-- ALWAYS-ON-DISCIPLINES:END -->

<!-- TELEGRAM-QUEUE:BEGIN (managed by .agents/scripts/sync-orchestration-sections.ps1 - edit the block there, not here) -->
## Queueing work for Telegram / Taco

"Queue this", "make a task", "tg/taco task", "add it to the queue", "save this for later" all mean ONE thing: an **Inbox task in the shared dashboard queue** (`cursor-dashboard/agent-os-state.json`, shared by every project), never a TODO in this repo or a note. The daemon is down until Trevor says otherwise, so do not probe it: one command writes the brief, adds the task to the file directly, and confirms it on a re-read:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\TrevorWhite\Downloads\Big Projects\.agents\scripts\add-dashboard-task.ps1" -Title '...' -Project 'kicker-analysis' -Owner ops -Brief '...'
```

**Queue it when effort is high, judgement is low, and "done" is statable** - if only Trevor can decide it that is a question, and if it is short just do it. `-Brief` is for a reader with no session context: real numbers, what done means, the command that verifies it. `owner` is the lane (`ops`, `taco` = phone lane, `research`, `codex`), **never Trevor**. Never hand-edit the state JSON; read and close tasks with `task-queue.py`. **Whoever works a task claims it first:** `task-queue.py start <id>` (`-Start` to queue and work it now). Trevor names several: one background agent per task at once, prompt only the id, brief path and shared-file limits; each claims its own. Full procedure: [`../.agents/rules/creating-dashboard-tasks.md`](../.agents/rules/creating-dashboard-tasks.md).
<!-- TELEGRAM-QUEUE:END -->

<!-- ORCHESTRATION-DEPTH:BEGIN (managed by .agents/scripts/sync-orchestration-sections.ps1 - edit the table there, not here) -->
## Orchestration & loops - execution depth

Choose **solo**, **subagents**, **workflow** or **loop** per [`../.agents/rules/orchestration-and-loops.md`](../.agents/rules/orchestration-and-loops.md) (reference: [`../.agents/memory/reference_dynamic-workflows.md`](../.agents/memory/reference_dynamic-workflows.md)). Separate doing from judging; state model tier, reasoning effort and cost before spawning. Loops need a hard stop and per-iteration checkpoint. Gates must be able to take DONE back.

**Anything needing judgement gets a skill and a second pair of eyes. Working from memory is the failure mode here.**

- **Use a fitting skill before starting.** If unknown, use `scout-skills` (`mid` | `high` | `max`). Fill real gaps with complete local bundles.
- **Independent verdicts.** Rankings, design choices, exploitability and numbers reaching people need a reviewer who did not produce them.
- **Breadth gets workers.** Three or more independent shards, each with its own proof: one worker per shard, then deterministic synthesis. Count actual work, including bundle fetch, mirrors and hashes, not just items.
- **Trivial work stays solo.** One-file mechanical edits need no worker. Cost gates still apply; judgement and breadth determine shape.

**Unattended Claude dispatch must prove progress.** New authorized launches use explicit `--permission-mode bypassPermissions`, owning project cwd and a permitted write location. Before launch or recovery, inspect saved permissions plus registered and detached runtime owners. Preserve existing session permissions; never reroute or restart around a real approval/isolation gate.

- **Accept evidence, not labels.** Within two minutes inspect the actual prompt. Launch stays unverified until the exact original receives the task, completes a task tool and advances an artifact/checkpoint. Pending initialization or approval is not running. Report a new actionable gate once.
- **Recover the original without copies.** Never `--resume --bg` a registered original, including idle or blocked sessions; use normal attachment. A live detached owner also forbids a second launch. Technical recovery needs diagnosed failure, no approval gate and one bounded exact-ID attempt. Honor stop/unmark/OFF and actual reset.
- **Pause, queue and gates outrank old continuation text.** Run `python .agents/scripts/dispatch-policy.py --session <full-id> --evidence <fresh-evidence.json>` from the workspace before recovery. Exit 2 means no dispatch. Check the latest allowlist first: no prompts before defer time and confirmed reset; manual approvals persist after reset; caller approval rejection is not an autonomous lane. Verify checkpoint/idle acknowledgment before saying paused. User holds require fresh continuation; disable reviewed completed tasks. Never retry approvals or route around a real gate.

**Here:** Weekly NFL kicker fantasy forecasts. Read README.md and METHOD.md. Preserve season-forward validation, negative findings and uncertainty. Separate engineering completion from claims of predictive edge.
<!-- ORCHESTRATION-DEPTH:END -->

