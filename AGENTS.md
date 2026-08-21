# RadioVue Agent Operating Guide

Read this file before performing meaningful work in this repository. It is the orchestration entry point: understand the task, select the applicable project knowledge, and then act. It does not replace detailed skills or specifications.

## Project Context

RadioVue is an experimental medical-imaging and machine-learning research project. Its current conceptual modules are:

- **Parallax**: one chest X-ray view to a complementary X-ray view.
- **Voluma**: two chest X-ray views to a 3D CT representation.
- **Recon**: generated CT to X-ray projection and comparison with the original X-ray.

These modules are research hypotheses and boundaries, not finalized architecture. Prefer the loop: research → hypothesis → experiment → evaluation → decision → implementation → documentation. Never present an assumption, visual result, or untested idea as an established result.

## Instruction Hierarchy

Resolve conflicts explicitly in this order:

1. System, platform, safety, and user instructions.
2. This `AGENTS.md` file.
3. Applicable OpenSpec project and module specifications.
4. Applicable skill instructions.
5. Repository documentation and module-specific instructions.

If instructions at the same level conflict, identify the conflict, use the more specific instruction when it does not violate a higher-level instruction, and ask for direction when the conflict materially changes scope or safety.

## Task Routing

Before meaningful action:

1. Understand the request and identify its work type, such as research, paper/repository/dataset analysis, DICOM processing, X-ray preprocessing, Parallax, Voluma, Recon, model experimentation, evaluation, debugging, architecture, documentation, experiment analysis, or project decision.
2. Inspect the relevant resources below and select the minimum applicable set; do not apply every skill by default.
3. Combine resources when needed, such as research plus a module specification plus experiment documentation.
4. Check for an existing relevant specification, research note, failed-experiment record, or decision record before creating new work.
5. Plan, execute, evaluate, record meaningful findings or decisions, and report what changed and why.

Do not start implementation merely because a request mentions a feature or model. First determine what project instructions and evidence apply. OpenSpec and skills support problem understanding; they do not replace it.

## Project Resources

- **OpenSpec project and module specifications:** `openspec/project.md` and `openspec/specs/`.
- **Codex skills:** `.codex/skills/`.
- **Claude skills and commands:** `.claude/skills/` and `.claude/commands/`.
- **Shared agent skills:** `.agents/skills/`.
- **Research Notes:** `research/notes/`.
- **Failed Experiment Log:** `research/failed-experiments/`.
- **Decision Log:** `research/decisions/`.

The three skill directories provide compatible entry points for different agents. Use the copy available to the active agent, and treat same-named skills as one shared workflow unless their contents differ.

Use `radiovue-research` for unfamiliar approaches or evidence gathering, `radiovue-experiment-log` after meaningful failed or abandoned experiments, and `radiovue-decision-log` after meaningful choices. Do not create a new skill if an existing one covers the work; if a durable capability is genuinely absent, identify the gap and recommend a new skill rather than improvising a permanent workflow.

## Research and Experimentation

For research or unfamiliar technical work, examine relevant papers, official repositories, implementations, weights, datasets, and limitations before proposing a major solution. Clearly label whether a statement is established research, repository behavior, experimental observation, project decision, hypothesis, or speculation.

Record meaningful failed attempts in the Failed Experiment Log with the attempt, rationale, outcome, evidence, lesson, and next experiment. Do not silently abandon evidence-rejected approaches.

Record meaningful architectural, model, data, preprocessing, evaluation, environment, or implementation choices in the Decision Log with alternatives, rationale, evidence, trade-offs, and reconsideration criteria. Do not log trivial edits as decisions.

## Medical-Research Safeguards

- Protect patient-identifying information and do not expose sensitive DICOM metadata.
- Treat generated images as research outputs, never clinical ground truth or clinical advice.
- Avoid unsupported medical claims and distinguish research use from clinical use.
- Preserve citations, licenses, environment details, configurations, data versions/splits, and quantitative evaluation where applicable.
- Document material assumptions and do not download, alter, or share medical data without explicit authorization.

## Completion

Adapt this workflow to the task’s scope. Before reporting completion, evaluate the result in proportion to its risk, update the relevant research or decision record when warranted, and state what changed, the evidence used, and any remaining uncertainty.
