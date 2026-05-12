# CLAUDE.md — xd-kb

## Dual VCS Workflow

This directory is tracked by both P4 and git:

- P4 path: `//XDrpg/develop/Tools/xd-kb/` (workspace root `.p4config`)
- Git remote: `https://github.com/zhing2006/xd-kb.git` (`master` branch)

**Every change must be both P4 submitted AND git committed + pushed**, in this order:

1. `p4 submit -d "<Chinese description>"`
2. `git add -A && git commit -m "<English description>"`
3. `git push` (for a release: `git tag vX.Y.Z && git push origin vX.Y.Z`)

## Version Bump

The crate version lives in **two** places that MUST stay in lockstep:

1. `Cargo.toml` (and `Cargo.lock` — `cargo` will rewrite this when it next runs, but bump it manually too so the locked version matches in the same commit)
2. `npm/xd-kb/package.json` — the npm wrapper's `version` field is used by `npm run pack:local` to derive the release tag and tarball name (`xd-kb-<version>.tgz`)

Bump both in the same change as the release tag (`vX.Y.Z`). A mismatched npm version will pack a tarball under the wrong filename and fetch the wrong GitHub release assets.

## Local Pack Output

`npm run pack:local` (run from `npm/xd-kb/`) writes the tarball to `../../../packages/` **outside** this repo, which resolves to the workspace path `Tools/packages/`. That directory is tracked by **P4 only** (it sits outside the git repo root). After each release, `p4 add` the new `.tgz` and submit it.
