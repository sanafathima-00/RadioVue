# RadioVue Project Specification

RadioVue is an experimental medical-imaging research project investigating whether limited chest X-ray information can support a useful whole-chest 3D CT representation. It is not a finalized product architecture and no current model, dataset, or paper is mandated.

## Modules

- **Parallax**: complementary chest X-ray view synthesis.
- **Voluma**: multi-view chest X-ray to 3D CT reconstruction.
- **Recon**: CT-to-X-ray projection and consistency evaluation.

## Research Rules

- Treat external claims as evidence only at their stated confidence and provenance.
- Keep hypotheses replaceable and make experiments reproducible.
- Evaluate anatomy and quantitative consistency, not visual plausibility alone.
- Maintain separate research notes, failed-experiment records, and decision records under `research/`.
- Do not place protected health information in specifications, logs, or prompts.
