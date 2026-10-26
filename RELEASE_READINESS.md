# MLX-TERNARY RELEASE READINESS ROADMAP

Master tracking checklist for open-source release readiness and PyPI distribution.

---

### Phase 1: Repo Hygiene & Security Sanitization
- [x] **1.1 Quarantine Sensitive Files:** Purged stray `.env`, `pantry/`, and `tracker.db` from git tree.
- [x] **1.2 Harden `.gitignore`:** Blocked `.env*`, `pantry/`, `*.db`, `*.sqlite`, `.DS_Store`, and `testing-artifacts/`.
- [x] **1.3 Bring Stranded Benchmarks In-Tree:** Migrated full benchmark harness into `benchmarks/` and updated `docs/benchmarks.md`.
- [x] **1.4 Fix Packaging & Editable Install:** Verified `pip install -e .` resolves `src/` out of the box with zero manual `PYTHONPATH` overrides.
- [x] **1.5 Contributor Guardrails:** Established `CONTRIBUTING.md` and public `GEMINI.md` enforcing zero-build pure-Python Metal JIT architecture.

---

### Phase 2: Standalone Benchmark Harness & Telemetry
- [ ] **2.1 Consolidated Benchmark Runner:** Build `benchmarks/run_benchmarks.py` capable of executing micro-kernel, layer projection, and end-to-end token decode sweeps with one command.
- [ ] **2.2 CLI Benchmark Entrypoint:** Wire `mlx_ternary bench` CLI into `src/mlx_ternary/cli.py`.
- [ ] **2.3 Hardware Verification on Apple Silicon:** Run full benchmarks on M-series hardware and verify the 1.42x speedup / 273 GB/s claims.
- [ ] **2.4 Structured Telemetry Output:** Ensure benchmark suite outputs structured Markdown tables and machine-readable `telemetry.json`.

---

### Phase 3: Testing Rigor & CI Automation
- [ ] **3.1 Boundary & Shape Invariance Fuzzing:** Add parameterized tests for non-power-of-2 dimensions ($M=7$, $K \neq 128n$, non-contiguous strides).
- [ ] **3.2 Memory Leak & JIT Idempotency Test:** Assert zero resident memory growth and zero recompilation overhead across 1,000 inference cycles.
- [ ] **3.3 GitHub Actions Workflow:** Scaffold `.github/workflows/ci.yml` running unit tests on macOS ARM64 runner (`macos-14` / `macos-15`).

---

### Phase 4: GitHub Showcase & Developer Onboarding
- [ ] **4.1 README Polish:** Embed the 5-second one-liner verification snippet and verified M4 Pro benchmark table.
- [ ] **4.2 Dead Link Audit:** Verify all links across `docs/` and `mkdocs.yml` resolve cleanly.
- [ ] **4.3 Minimal Runnable Example:** Add `examples/basic_inference.py` demonstrating zero-dependency 3-line inference.
