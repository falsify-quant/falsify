# Pre-launch checklist

Deferred decisions, recorded so they do not get lost between now and going public.
Nothing here blocks development; all of it blocks the first public link.

## Namespace

- [x] Claim a public namespace. `falsify` was already taken as a GitHub org.
      **`falsify-quant` claimed 2026-07-28**, matching the PyPI distribution name.
      Recorded for anyone repeating this: the GitHub API cannot tell you whether a name
      is claimable — it only ever proves a name is *taken* — so the web UI is the only
      real test.
- [ ] **Transfer** the repo into it, so it lands at `falsify-quant/falsify`. Do not rename
      the account instead: a username rename 404s every profile and gist link, and its
      repository redirects die the moment anyone claims the freed username and creates a
      matching repo name. Repo transfers are the sturdier mechanism, and a private repo
      has no inbound links to break.
- [x] Update the three places the URL is baked in: `pyproject.toml` (`[project.urls]`),
      the README badges and clone line, and the CONTRIBUTING clone line. All now point at
      `falsify-quant/falsify`. **These are ahead of the transfer** — the badge and the
      clone line are broken until it happens.

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
