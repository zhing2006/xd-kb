# xd-kb npm wrapper

This package installs the `xd-kb` command into npm's bin directory and forwards
all arguments to the platform-specific Rust binary from the GitHub Release.

## Create a local package

From the repository root:

```sh
npm --prefix npm/xd-kb run pack:local
```

The script downloads the five Release assets for the current git tag. If the
checkout is not exactly on a tag, it falls back to the package version tag
(`v0.1.0` by default). Assets are placed under `npm/xd-kb/vendor/`, and the
tgz is written to `../packages`.

Useful overrides:

```sh
XD_KB_RELEASE_TAG=v0.1.0 npm --prefix npm/xd-kb run pack:local
XD_KB_GITHUB_REPOSITORY=zhing2006/xd-kb npm --prefix npm/xd-kb run pack:local
XD_KB_PACK_DEST=../packages npm --prefix npm/xd-kb run pack:local
npm --prefix npm/xd-kb run fetch:release -- v0.1.0 --force
```

For a private GitHub repository, the script first tries `gh release download`,
so an existing `gh auth login` session is enough:

```sh
gh auth login
npm --prefix npm/xd-kb run pack:local
```

If `gh` is not installed, set `GITHUB_TOKEN` or `GH_TOKEN`. To force the API
path even when `gh` is available, set `XD_KB_NO_GH=1`.

## Install locally

```sh
npm install -g ../packages/xd-kb-0.1.0.tgz
xd-kb --version
```
