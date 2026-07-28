# Pre-launch checklist

Deferred decisions, recorded so they do not get lost between now and going public.
Nothing here blocks development; all of it blocks the first public link.

## Namespace

- [x] Claim a public namespace. `falsify` was already taken as a GitHub org.
      **`falsify-quant` claimed 2026-07-28**, matching the PyPI distribution name.
      Recorded for anyone repeating this: the GitHub API cannot tell you whether a name
      is claimable — it only ever proves a name is *taken* — so the web UI is the only
      real test.
- [x] **Transfer** the repo into it. Done 2026-07-28; it now lives at
      `falsify-quant/falsify`, still private, Actions carried over enabled. The account
      was deliberately *not* renamed instead: a username rename 404s every profile and
      gist link, and its repository redirects die the moment anyone claims the freed
      username and creates a matching repo name.
- [x] Update the three places the URL is baked in: `pyproject.toml` (`[project.urls]`),
      the README badges and clone line, and the CONTRIBUTING clone line. All now point at
      `falsify-quant/falsify`, and the transfer has caught up with them.

## PyPI

The name `falsify` belongs to an actively maintained project by Cüneyt Öztürk (MIT,
0.3.11, Python ≥3.11) — a CLI that pre-registers an ML evaluation claim as a SHA-256
manifest and later verifies PASS / FAIL / TAMPERED.

**It collides with us in two places, and an earlier note here said it did not.** Read from
the published wheel rather than the project description:

```
top_level.txt      falsify, falsify_prml, mcp_server
console_scripts    falsify       = falsify_prml:main
                   falsify-engine = falsify:main
```

- It ships a **top-level `falsify.py`**, so `import falsify` resolves to whichever of the
  two is earlier on `sys.path` when both are installed. Silent, and the failure is a
  confusing `AttributeError` rather than an `ImportError`.
- It owns the **`falsify` console script**. Installing both leaves one shadowing the
  other depending on install order.

Distribution names do not reserve module names, so nothing prevents shipping ours — the
question is whether to. Unresolved; see "The import-name decision" below.

- [ ] **Create the pending publisher on PyPI** — Account → Publishing → *Add a new pending
      publisher*. It is "pending" because the project does not exist yet; PyPI creates it
      on the first successful upload. Exact values:

      | Field | Value |
      |---|---|
      | PyPI project name | `falsify-quant` |
      | Owner | `falsify-quant` |
      | Repository name | `falsify` |
      | Workflow name | `publish.yml` |
      | Environment name | `pypi` |

      Do this *before* the first release. Registering the name by uploading with a token
      first, and adding trusted publishing afterwards, works but leaves a token in the
      account that then has to be remembered and revoked.
- [x] Trusted publisher rather than an API token in secrets. `.github/workflows/publish.yml`
      authenticates with a short-lived OIDC identity — `id-token: write` is the only
      permission it holds, and there is no secret to leak or rotate.
- [x] `python -m build && twine check dist/*` before the first upload. Verified clean: the
      wheel contains `falsify/` only, no `private/`, no `strategies/`, no `corpus/`.
- [ ] Attach a required reviewer to the `pypi` environment in repository settings.
      Without one the environment is only a label. A PyPI version number cannot be
      reused once uploaded — not even after deleting the file — so this is the last
      point at which a mistake is still free.

      **Now unblocked** — the repo is public, so protection rules are available on the
      free plan. Settings → Environments → `pypi` → Required reviewers.

### The import-name decision

Nothing blocks going public — this blocks the first *upload*, because the choice is
permanent the moment anyone installs it. Three options, in the order they are worth
considering:

1. **Keep `falsify`.** No churn, and the collision only bites the small set of people who
   install both an ML-eval pre-registration CLI and a backtest validator. But when it
   bites it is silent, and being the second package to claim a module name is a poor
   look for a project whose subject is rigour.
2. **Rename the module to `falsify_quant`, matching the distribution.** `import
   falsify_quant`, console script `falsify-quant`. Unambiguous forever, costs a
   mechanical rename across the package, the tests, the README and three entry points.
   Cheap now, expensive after anyone depends on it.
3. **Keep the module, rename only the CLI** (`falsify-quant` as the command). Halves the
   collision and leaves the silent half in place. Probably the worst of the three.

Decide before the first release, not after.

### The sdist has to be able to test itself

`twine check` validates metadata and says nothing about whether the artefact works. The
first build shipped `tests/test_canon.py` and `tests/test_corpus.py` without `strategies/`
or `corpus/`, so `pytest` on an unpacked tarball died at collection — while every CI job
stayed green. `MANIFEST.in` now excludes them, and CI unpacks the sdist somewhere else
entirely, installs it, and runs what it ships (288 tests), plus installs the wheel into a
clean virtualenv and calls all three console scripts.

## Flipping public — done 2026-07-28

The repository is public at `github.com/falsify-quant/falsify`.

- [x] `git ls-files | grep -c private/` printed **0** at the moment of the flip, and so
      did the stronger check: `private/` was never added in any of the 19 commits on any
      branch. A file deleted from `HEAD` is still in the history and still becomes public.
- [x] No credentials, keys, tokens or addresses in tracked content. Single author on every
      commit, no trailers.
- [x] **One real find, fixed before the flip:** the watch daemon's `CONFIG_EXAMPLE`
      carried a live bot's actual parameters — instrument, bar size, cutover date and both
      moving-average lengths — in a docstring inside the installed package. The audit is
      not a formality; run it against content, not just against paths.
- [x] CI green on all three operating systems. Six jobs: build, plus Python 3.10/3.12/3.13
      on Linux and 3.12 on macOS and Windows.
- [x] Read Öztürk's `falsify` on PyPI first. Done — see the PyPI section above for what
      it is and the two namespace collisions it creates. On the prior-art question the
      answer is reassuring: it locks a *claim* (metric, threshold, dataset hash, seed)
      **before** an experiment runs and later verifies the result against it, whereas
      `falsify/attest.py` signs a *finished verdict* and re-derives its arithmetic from
      the findings. Pre-registration versus post-hoc tamper-evidence — adjacent, not
      overlapping, and the two compose rather than compete.

## Launch content

- [x] The corpus study. Aggregate statistics only — **do not publish verdicts naming
      commercial vendors.** "Of N strategies, the median scored X, and one in six was
      reading data it could not have had" is the more interesting claim anyway, and it
      does not invite a defamation letter. Done: `corpus/FINDINGS.md`, 890 cells, no
      vendor named anywhere in the output.

## Licensing note

AGPL-3.0-or-later is deliberate: it permits everything except offering a modified falsify
to others *as a network service* without sharing source. Expect some firms to have a
blanket AGPL ban and to ask for a commercial licence. That is the intended outcome, not a
problem — the copyright is held solely by the author, so dual-licensing is available at
any time.
