# Pre-launch checklist

Deferred decisions, recorded so they do not get lost between now and going public.
Nothing here blocks development; all of it blocks the first public link.

## Namespace

The repo currently lives at `josephlangstroth-debug/falsify` because it is **private** and
the URL has no audience yet. That name is not the one to launch under.

- [ ] Claim a public namespace. `falsify` was already taken as a GitHub org.
      Candidates, in preference order: `falsify-quant` (matches the PyPI name),
      `falsifyhq`, `falsifylabs`, `getfalsify`. The GitHub API cannot tell you whether a
      name is claimable — it only ever proves a name is *taken* — so the web UI is the
      only real test.
- [ ] **Transfer** the repo into it. Do not rename the account instead: a username rename
      404s every profile and gist link, and its repository redirects die the moment
      anyone claims the freed username and creates a matching repo name. Repo transfers
      are the sturdier mechanism, and a private repo has no inbound links to break.
- [ ] Update the three places the URL is baked in: `pyproject.toml` (`[project.urls]`),
      the README badges and clone line, and the CONTRIBUTING clone line.

## PyPI

- [ ] Register **`falsify-quant`**. The name `falsify` belongs to an actively maintained
      project (Cüneyt Öztürk, 17 releases, last 2026-07-17). Its import name is
      `mcp_server`, so `import falsify` and the `falsify` CLI are unaffected — only the
      distribution name differs.
- [ ] Set up a PyPI trusted publisher for the repo rather than an API token in secrets.
- [ ] `python -m build && twine check dist/*` before the first upload. Already verified
      clean: the wheel contains `falsify/` only, no `private/`, no `strategies/`.

## Before flipping public

- [ ] `git ls-files | grep -c private/` must print **0**. This is the one that matters —
      `private/` holds live trading parameters, traded instruments, position sizing and
      paths to databases with real account activity.
- [ ] CI green on all three operating systems.
- [ ] Read Öztürk's `falsify` on PyPI first. It pre-registers ML evaluation claims as
      SHA-256 hashes — the same pre-commitment idea as the planned attested reports,
      applied to ML evals. Prior art worth understanding before designing ours.

## Launch content

- [ ] The corpus study. Aggregate statistics only — **do not publish verdicts naming
      commercial vendors.** "Of N strategies, the median scored X, and one in six was
      reading data it could not have had" is the more interesting claim anyway, and it
      does not invite a defamation letter.

## Licensing note

AGPL-3.0-or-later is deliberate: it permits everything except offering a modified falsify
to others *as a network service* without sharing source. Expect some firms to have a
blanket AGPL ban and to ask for a commercial licence. That is the intended outcome, not a
problem — the copyright is held solely by the author, so dual-licensing is available at
any time.
