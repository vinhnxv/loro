"""Make-style artifact layer: durable per-stage artifacts with fingerprint
sidecars, unified retry policy, skip ledger, preflight, QA gate and run report.

Nodes keep their (state, cfg) -> dict signature; the harness wraps their
expensive work in load-or-compute so a rerun of the same command resumes
instead of recomputing.
"""
