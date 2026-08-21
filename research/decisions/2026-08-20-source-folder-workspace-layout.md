# Source-Folder Workspace Layout Decision

## Decision

Use one anonymized `Patient_###` workspace folder per source folder containing an IM0 image, with an explicitly requested default file name of `PA.jpg`.

## Date

2026-08-20

## Context

The source dataset contains one IM0 image in each of 273 non-empty source folders, while its DICOM metadata identifies the sources as one patient and does not declare a view position.

## Options Considered

- Group output by DICOM PatientID and use generic image names.
- Use each source folder as a non-clinical workspace unit and retain generic image names.
- Use each source folder as a non-clinical workspace unit and apply the user-provided `PA` default name.

## Chosen Approach

Create one flat `Patient_###/PA.jpg` workspace unit per source folder.

## Why We Chose It

The folders are intended to hold an input X-ray and its later generated complementary view together, rather than to represent clinical patients or an automated view classification.

## Evidence and Reasoning

A three-source test produced three separate anonymized folders containing verified `PA.jpg` files, and the raw dataset was unchanged during conversion.

## Trade-offs

The output folder name is a workspace identifier rather than a clinical patient identity, and `PA` is a user-provided default label that must be manually changed if a lateral view is found.

## What Would Cause Reconsideration

Revisit this layout if source folders begin to contain multiple unrelated studies, if input and generated views require a more explicit manifest, or if reliable view metadata becomes available and an automated naming policy is desired.
