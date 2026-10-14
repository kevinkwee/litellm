# AGENTS.md

Guidance for AI coding agents working in this repository. `CLAUDE.md` and `GEMINI.md` are thin pointers to this file, so this file is the single source of truth for agent conventions.

## Project Overview

LiteLLM is a Python SDK and OpenAI-compatible proxy gateway that gives one uniform interface to 100+ LLM providers. The repo holds the Python package (`litellm/`), a Rust workspace (`litellm-rust/`) with three crates (pure transforms in core, the axum-based ai-gateway, and a PyO3 bridge to the Python SDK), a Next.js admin dashboard (`ui/litellm-dashboard/`), and CI and infrastructure configuration.

## Prerequisites

- Python `>=3.10, <3.14` (details in Language and Runtime Constraints). `uv` drives the lint and test environments through the Makefile.
- `make install-test-deps` installs the proxy dev environment plus the test tooling; the install targets also generate the Prisma client that basedpyright expects to resolve (`litellm/proxy/schema.prisma` is the schema).
- Node and npm for the dashboard (`ui/litellm-dashboard`); the dev server is `npm run dev` on port 3000.
- API keys and other secrets come from `.env`. Reading them for local runs is fine; their values must never enter conversation history, commits, GitHub issues, or PRs.

## Project Map

| Folder | Purpose | Precedence file |
|--------|---------|-----------------|
| `litellm-rust/` | Rust workspace: pure transforms (core), I/O gateway (ai-gateway), PyO3 binding (python-bridge) | [`litellm-rust/AGENTS.md`](litellm-rust/AGENTS.md) |
| `litellm/proxy/_experimental/mcp_server/` | MCP server mounted on the proxy | [`litellm/proxy/_experimental/mcp_server/AGENTS.md`](litellm/proxy/_experimental/mcp_server/AGENTS.md) |
| `ui/litellm-dashboard/src/components/chat/` | Chat UI components | [`ui/litellm-dashboard/src/components/chat/AGENTS.md`](ui/litellm-dashboard/src/components/chat/AGENTS.md) |
| `tests/e2e/` | End-to-end harness | [`tests/e2e/CLAUDE.md`](tests/e2e/CLAUDE.md) |

When working inside a mapped folder, read and follow that folder's file in addition to this one. Folder-level files win for module-specific rules; this file wins for cross-cutting rules such as runtime versions, security, and testing philosophy.

## Core Philosophy

### Trade-off Priority (when conflicts arise)

Project order:

1. **Correct**: the code does what it should
2. **Secure**: the code adds no attack surface
3. **Performant**: the code is fast enough
4. **Readable**: future readers understand it
5. **Maintainable**: the code is easy to maintain and change
6. **Modern**

### Ground Rules

- Read and understand existing code before modifying it. Treat the code as a starting point, not as evidence of correctness. Legacy code carries bad practices, over-complex code, and code smells. If something looks wrong, speak up and write new code the right way (correct, secure, performant, readable, maintainable, modern) instead of propagating the problem, even when that means departing from the surrounding code.
- Isolate new code from legacy where possible, and never copy legacy code into new code without reviewing its quality.
- **Think before coding.** State your assumptions. When uncertain, ask. When multiple interpretations exist, present them instead of picking silently. If a simpler approach exists, say so and push back when warranted. If something is unclear, stop and name the confusion.
- **Simplicity first.** Write the minimum code that solves the problem, nothing speculative. No features beyond what was asked, no abstractions for single-use code, no configurability that was not requested, no error handling for impossible scenarios. If you write 200 lines and it could be 50, rewrite it. Ask whether a senior engineer would call the code overcomplicated.
- Prefer incremental delivery: core logic first, then edge cases, then refinements.
- Do not break foundation code. Changes to widely-depended-on modules ripple into every dependent, so verify dependents before changing a shared field, method, or contract.
- If a requirement is ambiguous, ask before writing code.

## Design Principles

### SOLID

| Principle | Rule | Common Mistake |
|-----------|------|----------------|
| **SRP** (Single Responsibility) | Each class/function/file has ONE reason to change. If you need "and" or "or" to describe it, split it. | Interpreting SRP as "one function per class." SRP means one *axis of change*. |
| **OCP** (Open/Closed) | Add new behavior by writing new code, not modifying existing code. Use polymorphism or strategy patterns where change is expected. | Over-engineering with premature abstractions. Apply OCP where you have *evidence* of changing requirements. |
| **LSP** (Liskov Substitution) | Subclasses must honor the contract of their parent. Prefer composition over inheritance when "is-a" is not strict. | Overriding a method to throw `NotImplementedError` or do nothing. |
| **ISP** (Interface Segregation) | Define small, role-specific interfaces. Clients depend only on methods they use. | Creating one "service" interface stuffed with methods for many unrelated roles. |
| **DIP** (Dependency Inversion) | Depend on abstractions at module boundaries, not concrete implementations. | Confusing DIP with "just use dependency injection." DIP is about inverting the *direction of source-code dependency*. |

### DRY (Don't Repeat Yourself)

- Extract duplicated logic when the copies are the same business rule and the duplication already costs more than a shared implementation would.
- Single source of truth for configuration, constants, and schema definitions.
- **Prefer duplication over wrong abstraction.** Two pieces of code that look similar but serve different business purposes are NOT duplication, because merging them creates accidental coupling.
- "Wrong abstraction" means: premature generalization, unclear purpose, or coupling unrelated concerns.

### KISS (Keep It Simple)

- Choose the simplest implementation that satisfies current requirements.
- Prefer standard library solutions over custom implementations.
- A plain function call beats metaprogramming. A dictionary beats a class when all you need is data grouping.
- Do not add design patterns, abstractions, or frameworks "just in case."

### YAGNI (You Aren't Gonna Need It)

- Implement features only when there is a concrete, current requirement.
- Do not build generic/extensible frameworks before you have at least two concrete use cases.
- Delete speculative code and unused feature flags regularly.
- Three similar lines of code is better than a premature abstraction.

## Language and Runtime Constraints

Python is pinned to `>=3.10, <3.14` in `pyproject.toml`. `match`/`case`, PEP 604 unions (`int | None`), builtin generics (`list[int]`), and the walrus operator are all available. Do not use anything that only works on 3.14 or later. For Rust and the dashboard, follow the crate- and folder-level AGENTS.md files and their toolchains.

## Code Quality

### Comments and Docstrings

- **Do not write any comments** (existing comments stay untouched) **unless the user explicitly asks.** When a comment is requested or you are editing an existing one, the rules below apply.
- Comments explain **why**, never **what**. If a better name or a better function removes the need for a comment, do that instead.
- A docstring is a concise summary of the unit's purpose and role, not its wiring. Restating signatures, type hints, decorator arguments, or config keys is redundancy, not a summary.
- Every comment and docstring must make sense to a cold reader: no change narration ("previously", "was removed", "moved to"), no plan/ticket/iteration references, no section separator banners, no commented-out code, no TODOs without a findable ticket, and no cross-repo file paths.

### Naming

- Use intention-revealing names. Functions start with a verb; classes are nouns; booleans read as a yes/no question (`is_`, `has_`, or a plain adjective when the context makes the type clear).
- Framework-dictated names (overrides, interface implementations) stay as they are.
- No abbreviations beyond universally understood ones (`id`, `url`, `api`), and no generic names (`data`, `result`, `obj`, `thing`, `temp`, `misc`, `utils`).
- "And"/"or" in a unit's name signal two responsibilities: usually a reason to split. When the joined form is genuinely one domain concept, the name is fine.

### Typing, Immutability, and Structure

These rules apply to new and updated code. A three-line fix in a legacy file must never trigger a large drive-by refactor.

- Composition over inheritance.
- Never-nester: guard clauses and early returns, with the happy path at the lowest nesting level.
- Do not throw. Model failures as values, and map error unions to the existing public exception contracts with an exhaustive `match` plus `assert_never` (the pattern of `raise_public`).
- No mutation. Do not reassign variables, global or local. Instead of mutable lists and dicts, prefer tuples, frozen dataclasses with `slots=True`, and one-shot constructions with comprehensions or generators. On an LIT001 or LIT002 failure, refactor toward the functional style instead of introducing mutable structures; reach for `# mutable-ok` only as a genuine last resort when an immutable rewrite is truly impossible, and always pair it with a real reason.
- Use dependency injection, and do not monkeypatch class attributes for tests. Pass mocked dependencies in, so unit tests do not need patching.
- Fully typed. No `Any`, no coarse types like `dict[str, Any]`, and every function parameter strongly typed. When a new function would need untyped values, validate at the caller with Pydantic (a model or `TypeAdapter` that returns the typed thing or raises) and pass the typed variable in, instead of pushing `reportAny` / `reportExplicitAny` against their budget ceilings.
- Tagged unions matched with `match`.
- No monster files or god objects, and no file sprawl: keep a deliberate file and folder structure.
- Standard over hand-rolled: the official SDK or a library where one exists; industry standards instead of local conventions.
- API-fragmentation-aware: when logic must branch on which API surface produced or consumes data (chat completions, Anthropic Messages, Responses API shapes), check for an existing shared helper in `litellm_core_utils/prompt_templates/factory.py` first. If none exists, add one there instead of duplicating the same format-detection logic in every new guardrail or integration.
- Python's max line length is 120, not 88.

### Linting, Formatting, and Type Discipline

- Ruff lints and formats; basedpyright checks types. Three budget files gate their ceilings: `ruff-strict-budget.json`, `type-discipline-budget.json`, and `basedpyright-code-budget.json`.
- Run the lint targets on the files you changed (`make lint-dev`, or the narrower `lint-*` targets). CI parity is `make lint-checks`.
- When you fix violations gated by a budget file, run `make lint-budget-update` and commit the lowered limits so the ceilings ratchet down instead of leaving stale headroom. It measures the working tree, so the tree must contain exactly the fixes being committed.
- Run `make pre-commit` right before each commit; it generates types as needed, then formats and lints. Fix every error it reports. It only runs when staged frontend and/or backend changes exist, so stage your commit scope first (or stash/delete unwanted files in `litellm/` or `ui/`, where the backend and frontend lint run). If it fails because dashboard API types are stale, it already regenerated them: stage `schema.d.ts`, rerun `make pre-commit`, then commit.
- Every lint or type suppression must name the exact rule inside brackets and carry a reason, e.g. `# pyright: ignore[reportArgumentType]  # stubs lack async overload` or `# noqa: TID251  # <reason>`. `# type: ignore` is banned (LIT009): `pyrightconfig.json` sets `enableTypeIgnoreComments` to false, so it silently does nothing.
- When the linter reports a violation, take the least invasive option that keeps the rule meaningful, in this order: fix the code; configure the rule (when its intent is sound but the defaults do not fit); per-line suppression with the exact rule codes; block-scope suppression with a matching re-enable; narrowly-scoped per-file ignores in the config; global ignore only as a last resort, with a documented reason.

## Architecture

```
litellm/                  # Python SDK and proxy
├── llms/                 # one module per provider (chat, embedding, responses API, ...)
├── litellm_core_utils/   # shared core; prompt_templates/factory.py owns cross-API transforms
├── proxy/                # FastAPI proxy server: auth, routing, budgets, admin endpoints
│   └── schema.prisma     # proxy database schema
└── router.py             # load balancing, retries, and fallbacks across deployments

litellm-rust/             # three crates: core (pure transforms), ai-gateway (I/O), python-bridge (PyO3)
ui/litellm-dashboard/     # Next.js admin UI
tests/test_litellm/       # unit tests mirroring litellm/
tests/e2e/                # live end-to-end suite
```

This structure evolves with the codebase. If you add new modules, update this diagram and the corresponding sections.

### Separation of Concerns

- Keep domain logic free of framework, database, and HTTP imports; keep side effects (I/O, logging, metrics) at the edges.
- The Rust crate dependency direction is acyclic: core, then ai-gateway, then python-bridge. A default to a module over a new crate is a real trigger only for separate artifacts, proc macros, shared foundation, or publishable standalone code (details in `litellm-rust/AGENTS.md`).

## Security

- Sanitize and validate all user and external input at the boundary. Never trust data from outside the system boundary.
- Allowlists over denylists: reject by default, accept only known-good patterns.
- Use schema validation libraries (the codebase standard is Pydantic); do not hand-roll validation for complex structures.
- Keep secrets out of code: environment variables or secret managers, never hardcoded keys or tokens. `.env` values back local runs, but they must never reach conversation history, commits, GitHub issues, or PRs.
- SQL queries use parameterized statements, never string concatenation.
- Enforce access control on every new resource with least privilege. Do not make features universally accessible, and re-check access rules when touching existing resources.
- Do not expose internal details in error messages intended for end users.
- CI supply-chain safety: never pipe a remote script into a shell (`curl ... | bash`, `wget ... | sh`). Download the artifact to a file, verify its SHA-256 checksum against the provider's official `.sha256` / `.sha256sum` sidecar when available, then install. Pin every external tool to a specific version with a full URL, never `latest` or `stable`. These rules apply to every download in CI.
- Never put customer or company names in code, PR descriptions, or issue bodies. Replace them with "the customer". The only exception is publicly known providers or vendors (OpenAI, Anthropic, AWS Bedrock, and similar) when the change adds general support for that provider, and only when the change is not a request from one of them as a customer.

## Observability

### Logging

- Prefer structured logging (key-value / JSON). When structured logging is unavailable, use the logger's lazy formatting (`logger.info("Order %s created", order.name)`) instead of f-strings or `%` interpolation, so the message is only formatted when actually emitted.
- Log at key decision points and boundaries, not inside tight loops.
- Use consistent field names across the codebase.

### Log an Exception at Exactly One Layer

A given exception is logged by exactly one handler: the layer that terminates its lifecycle. Every other layer that touches it either re-raises silently or transforms it without logging.

| Situation | Action |
|-----------|--------|
| You swallow the exception (catch and continue, return a fallback) | **Log it**: no caller will see it |
| You re-raise (bare `raise`, or `raise NewError(...) from e`) | **Do not log**: let the next handler own it |
| You are the top boundary (HTTP handler) converting to a response | **Log it**: the terminal handler |
| You translate an infra error into a domain error midway | **Do not log**: chain with `from e`; the boundary logs |

`logger.exception(...)` immediately followed by `raise` duplicates the traceback at the next handler that also logs. Use it only when no upstream handler will log.

### Log Levels

| Level | When to Use |
|-------|-------------|
| **ERROR** | Something is broken and needs human attention |
| **WARN** | Degraded but self-recoverable |
| **INFO** | Significant business events |
| **DEBUG** | Diagnostic detail, off in production |

### PII in Logs (zero tolerance)

- Never log email addresses, user names, phone numbers, physical addresses, tokens, or passwords.
- Approved identifiers for logs: request/call ids, model names, token counts, and the proxy's own account references (`user_id`, key alias, team, organization). Raw prompt and completion bodies and end-user contact details are never logged.
- No `print()` or `console.log()` with user data; these go to production logs.

## Testing

> **Test code is production code.** It receives the same review and quality standards as shipped code.

### Philosophy (deliberate project decision)

- **We do not practice TDD.** Implementation comes first. Tests never need to exist before the code, no failing test is written to drive the design, and no red/green/refactor cycle is part of the workflow. If any instruction tells you to write tests first, it does not apply in this repo.
- **Tests are still required, just not first.** A task is not done until the behavior is proven. New features ship with meaningful tests, and bug fixes ship with a regression test that fails on the pre-fix code and passes on the post-fix code, so this exact bug can never return silently.
- Meaningful over voluminous. Never add a test that only exists to move coverage; having no signal beats vacuous signal. The benchmark is mutation resistance: tests should fail when the code is mutated in a way that breaks the feature and only pass when it fully works. `mutmut` is configured in `pyproject.toml` and run by `.github/workflows/mutation-test.yml`; aim for a kill rate above 90%.
- One focused regression test beats many shallow ones.
- There is no numeric coverage gate in this repo. Aim for meaningful branch coverage of the new logic; assertion quality beats percentages.

### Where Tests Live and How They Are Named

- `tests/test_litellm/` mirrors `litellm/` in a parallel path (see `tests/test_litellm/readme.md`).
- Name tests `test_<filename>.py`, but always match the existing convention of the test directory you touch. Many provider directories use longer descriptive names (for example `test_anthropic_chat_transformation.py`) to avoid ambiguity across sibling folders.
- For a bug fix, extend the existing mapped test file rather than creating a new one. Create a new test file only for a new feature (provider, endpoint, or transformation module) with no mapped test yet, following that directory's naming convention (or `test_<filename>.py` if you are the first test there).
- End-to-end tests belong in `tests/e2e/` and must follow the harness conventions documented in [`tests/e2e/CLAUDE.md`](tests/e2e/CLAUDE.md).

### How to Run Tests

- Install test deps once per clone: `make install-test-deps`.
- Touched modules: `uv run pytest tests/test_litellm/<mapped test path> -x -vv`.
- Suite slices matching the CI job groups: `make test-unit-llms`, `make test-unit-proxy-core`, `make test-unit-proxy-guardrails`, `make test-unit-proxy-misc`, or the full unit suite `make test-unit`.
- pytest runs with `asyncio_mode = "auto"`; honored markers include `limit_leaks` and `no_parallel`.
- Run the mapped tests before you commit, and fix all failures.

### How Tests Are Written

- Arrange-Act-Assert, one act per test, one logical assertion per test. Tests are independent, deterministic, and order-agnostic.
- Mock or fake external dependencies (DB, APIs, filesystem, time, randomness); never hit real services in unit tests. When a mock must raise, it raises a concrete type (the domain error the real code would raise), and raise-assertions such as `pytest.raises` pin the specific expected type, so a broader regression error cannot slip through.
- Test behavior through public APIs rather than private internals.
- Happy path first, then adversarial cases: unexpected input types (`None`, `""`, `[]`, `{}`, `0`, `-1`), boundary values, malformed data, error states from failing dependencies, and states that must be rejected. Error behavior checks include how it fails, not just that it fails.
- Write tests from the requirements, not from what the source code currently does. When a test fails, first ask whether the code is wrong; never bend a failing assertion to match current behavior without understanding why.

### What NOT to Test

- Simple getters, setters, and trivial mappers.
- Implementation details such as call order or internal state; test behavior instead.
- Coverage inflation with meaningless assertions.

### Anti-Patterns (forbidden)

| Pattern | Problem |
|---------|---------|
| **The Liar** | Test passes but does not verify the behavior it claims to test |
| **The Mirror** | Test reads the source and asserts exactly what the code does, finding zero bugs |
| **The Giant** | 50+ lines of setup, multiple acts, dozens of assertions; should be several tests |
| **The Mockery** | So many mocks that the test only tests the mock setup |
| **The Inspector** | Coupled to implementation details, breaks on any refactor |
| **The Chain Gang** | Tests depend on execution order or share mutable state |
| **The Flaky** | Passes and fails with no code changes |

## Pull Requests and Commits

- Conventional Commits for commit names and PR titles.
- Branch flow: branch off `litellm_internal_staging` and prefix the branch with `litellm_`. Never use a `claude/` prefix, never a `/` in branch names, and never set the PR base to `main`; `litellm_internal_staging` serves that purpose.
- Before writing a PR body, read `.github/pull_request_template.md` from disk and treat the comments and imperative instructions inside it as rules to follow. Agent harnesses may strip HTML comments from injected copies, so the disk copy is authoritative.
- In the "Linear ticket" section, write "Resolves LIT-1234" only with the real ticket id; otherwise leave the section blank. Never make up or search for a ticket id.
- Keep the PR description in sync with new commits. For related follow-up work, commit directly to the existing PR's branch; do not open a new branch or PR off the existing PR.
- Proof of fix: never pytest output or mock-only runs. Show a real end-to-end run that hits real provider APIs at real cost, using the same commands an end user would run. Curl a live proxy on `localhost:4000`:

  ```bash
  python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
  ```

  For bug fixes show the reproduction before the fix and the passing behavior after, with the commit hash of each run. For UI work, list the URLs (e.g. `http://localhost:4000/ui/?page=logs`), clicks, and fields in order so the user can run them and post screenshots. For the preferred compact style (a `for` loop of curl commands rather than throwaway scripts), see PR #27703.
- Tone rules for public-facing text (PR descriptions, comments, issues, commit messages): no emojis; no em-dashes anywhere; no "It's not X, it's Y" constructions; prose over bullets and numbered lists unless a list is the only sensible form; no trailing "." at the end of a paragraph when new lines follow; and avoid arrows, using `->` when a pointer is truly needed.
- Commit and push your work when you are done without asking.
- No AI attributions: no `Co-Authored-By: Claude` or similar trailers in commit messages, and no "Generated with ..." lines in PR descriptions or comments.

## Code Review

### Priority (blockers first)

1. **Security & PII**: no PII in logs, no hardcoded secrets, input validation
2. **DRY**: no duplicate types, classes, functions, or logic
3. **File Structure**: single responsibility per file, naming respected
4. **Architecture**: single responsibility, proper layer separation
5. **Code Quality**: SOLID, strong typing, error handling
6. **Testing**: meaningful regression tests that would fail if the logic broke, covering the happy path and adversarial cases
7. **Observability**: structured logging, no PII

Review questions for tests: would these tests catch a regression if someone broke the logic; are there untested edge cases or failure modes; if a line of business logic is removed, does at least one test fail.

### Legacy Code

- Do not prolong bad patterns; even surrounded by bad code, write good code.
- Do not copy from legacy code without reviewing quality.
- Isolate new code from legacy where possible.

## Documentation

Update `README.md` when you add new features, endpoints, or significant changes. Update this file (`AGENTS.md`) when you add new modules, classes, or change the architecture. Both must stay in sync with the codebase.

## Key Concepts

- **Model metadata**: prices, limits, and identifiers live in `model_prices_and_context_window.json`. When referencing or running models (coding, QA-ing, writing docs or tests), use the latest model of that family; treat training knowledge, memories, configs, and tests as stale, and determine the family's latest from `model_prices_and_context_window.json` or the web.
- **Cross-API transforms**: `litellm_core_utils/prompt_templates/factory.py` is the shared layer for converting between chat-completions, Anthropic Messages, and Responses API shapes.
- **Proxy database**: `litellm/proxy/schema.prisma` is the schema; the install targets regenerate the Prisma client, so basedpyright resolves the same modules CI does.

## Rule of Thumb

**When in doubt, choose simplicity. When trade-offs arise, follow the priority order in [Core Philosophy](#core-philosophy).**
**Build for correctness first. Optimize later. Prove the behavior with tests before you ship.**
