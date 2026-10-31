# mixing2006 Skills

Repo-local skills live under `skills/`. Use them when a task clearly matches
the described workflow.

## Available Skills

- `physics-validation`
  - file: `skills/physics-validation/SKILL.md`
  - use for changes to `na49_mixing/`, decay chains, event mixing,
    fragmentation-side logic, or benchmark physics behavior
- `example-smoke-test`
  - file: `skills/example-smoke-test/SKILL.md`
  - use for `examples/`, README command validation, packaging/path fixes, and
    small artifact-generation checks
- `sps-paper-pipeline`
  - file: `skills/sps-paper-pipeline/SKILL.md`
  - use for `sps_resonances_paper/`, constrained-fit wrappers, pipeline
    environment variables, and staged paper reproduction runs

## Usage Notes

- Start with one skill. Combine skills only when the task spans multiple
  surfaces, such as a physics change that also modifies example output.
- Prefer the smallest command that proves the change.
- Keep heavy pipeline stages opt-in. Use targeted tests, `--help`, single-stage
  runs, or dry-run wrappers before broader campaigns.
