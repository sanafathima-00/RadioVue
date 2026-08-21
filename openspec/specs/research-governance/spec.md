# Research Governance Specification

## Purpose

Maintain traceable, reproducible, evidence-based RadioVue experimentation.

## Requirements

### Requirement: Evidence provenance

Research claims SHALL distinguish published papers, official code, third-party code, released weights, experimental results, and internal hypotheses.

### Requirement: Separate records

Research notes, failed-experiment entries, and decision entries SHALL remain in their respective `research/` subdirectories and SHALL not be used interchangeably.

### Requirement: Reproducibility

Meaningful experimental results SHALL record code revision, environment, configuration, input data version/split, random seed where applicable, metrics, and output location.
