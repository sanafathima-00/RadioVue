# X-ray Working-Copy Preprocessing Decision

## Decision

Create a separate JPEG working copy from decodable DICOM sources named `.IM0` or extensionless `IM0`, while leaving the raw dataset untouched.

## Date

2026-08-20

## Context

The inspected raw dataset uses 273 extensionless files named `IM0`; sampled files contain a DICOM preamble and decode as 16-bit monochrome chest radiographs.

## Options Considered

- Trust the `.IM0` extension alone.
- Treat every extensionless file as an image.
- Accept both `.IM0` names and the observed extensionless `IM0` convention, then strictly validate each file as DICOM.

## Chosen Approach

Accept the documented and observed source-name conventions, then use strict DICOM decoding and require a non-empty DICOM PatientID for in-memory patient grouping.

## Why We Chose It

The observed dataset does not use a `.IM0` extension, so extension-only discovery would omit valid sources; accepting all extensionless files would risk processing unrelated files, while strict DICOM decoding and a required grouping identifier prevent both assumptions.

## Evidence and Reasoning

A five-source sample decoded successfully as 16-bit `MONOCHROME1` DICOM images with DICOM window information, grouped into one anonymized patient folder, and produced verified JPEG outputs without altering the raw-tree fingerprint.

## Trade-offs

Files with a missing PatientID stop conversion even if their pixels decode, and JPEG provides a display-oriented working copy rather than preserving diagnostic DICOM fidelity.

## What Would Cause Reconsideration

Revisit this policy if a later source collection uses a different reliable grouping identifier, contains valid non-DICOM medical image formats, or requires a lossless/quantitative working format in addition to JPEG.
