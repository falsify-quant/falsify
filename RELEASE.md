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

The name `falsify` belongs to an actively maintained project (Cüneyt Öztürk, 17 releases,
last 2026-07-17). Its import name is `mcp_server`, so `import falsify` and the `falsify`
CLI are unaffected — only the distribution name differs.

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

### The sdist has to be able to test itself

`twine check` validates metadata and says nothing about whether the artefact works. The
first build shipped `tests/test_canon.py` and `tests/test_corpus.py` without `strategies/`
or `corpus/`, so `pytest` on an unpacked tarball died at collection — while every CI job
stayed green. `MANIFEST.in` now excludes them, and CI unpacks the sdist somewhere else
entirely, installs it, and runs what it ships (288 tests), plus installs the wheel into a
clean virtualenv and calls all three console scripts.

## Before flipping public

- [ ] `git ls-files | grep -c private/` must print **0**. This is the one that matters —
      `private/` holds live trading parameters, traded instruments, position sizing and
      paths to databases with real account activity. Currently 0; re-check at the flip,
      not once.
- [x] CI green on all three operating systems. Six jobs: build, plus Python 3.10/3.12/3.13
      on Linux and 3.12 on macOS and Windows.
- [ ] Read Öztürk's `falsify` on PyPI first. It pre-registers ML evaluation claims as
      SHA-256 hashes — the same pre-commitment idea as the planned attested reports,
      applied to ML evals. Prior art worth understanding before designing ours.

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
