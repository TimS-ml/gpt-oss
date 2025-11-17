"""
Evaluation Framework Module

This module provides a comprehensive evaluation framework for testing language models
on various academic and professional benchmarks. The framework includes:

- Multiple evaluation benchmarks (GPQA, AIME, HealthBench)
- Flexible sampler backends (Chat Completions API, Responses API)
- Automated grading and scoring systems
- HTML report generation for detailed analysis

The module can be run from the command line using:
    python -m gpt_oss.evals --model <model_name> --eval <eval_name>
"""
