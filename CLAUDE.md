# CLAUDE.md — xd-kb

## Dual VCS Workflow

This directory is tracked by both P4 and git:

- P4 path: `//XDrpg/develop/Tools/xd-kb/` (workspace root `.p4config`)
- Git remote: `https://github.com/zhing2006/xd-kb.git` (`master` branch)

**Every change must be both P4 submitted AND git committed + pushed**, in this order:

1. `p4 submit -d "<Chinese description>"`
2. `git add -A && git commit -m "<English description>"`
3. `git push` (for a release: `git tag vX.Y.Z && git push origin vX.Y.Z`)

## Local Pack Output

`npm run pack:local` (run from `npm/xd-kb/`) writes the tarball to `../packages/` **outside** this repo, which resolves to the workspace path `Tools/packages/`. That directory is tracked by **P4 only** (it sits outside the git repo root). After each release, `p4 add` the new `.tgz` and submit it.
