# Amazon Business Entity Resolution Challenge

Team implementation for the Business Entity Resolution Challenge.

## Problem

Match every Source 1 business entity with all corresponding Source 2
and Source 3 entities using only the supplied challenge data.

## Planned Architecture

Normalization
→ Multi-pass Blocking
→ Lexical Retrieval
→ Semantic Retrieval
→ Candidate Pruning
→ Pairwise Feature Engineering
→ LightGBM Matcher
→ F0.5 Calibration
→ Singleton Decision
→ Final Outputs

## Repository Structure

student_resource/
    dataset/
    utils/
    Documentation_template.md

src/
experiments/
models/
output/

## Team Workflow

Do not work directly on `main`.

Create a feature branch for your work and open a pull request.