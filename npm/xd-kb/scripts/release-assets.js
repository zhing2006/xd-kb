const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const ASSETS = [
  { file: "xd-kb-win32-x64.exe", executable: false },
  { file: "xd-kb-linux-x64", executable: true },
  { file: "xd-kb-linux-arm64", executable: true },
  { file: "xd-kb-darwin-x64", executable: true },
  { file: "xd-kb-darwin-arm64", executable: true },
];

function packageRoot() {
  return path.resolve(__dirname, "..");
}

function readPackageJson(root = packageRoot()) {
  return JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));
}

function defaultTag(root = packageRoot()) {
  const tag = currentGitTag(root);
  if (tag) {
    return tag;
  }

  return `v${readPackageJson(root).version}`;
}

function currentGitTag(root = packageRoot()) {
  const candidates = [path.resolve(root, "..", ".."), root];

  for (const cwd of candidates) {
    const result = spawnSync("git", ["-C", cwd, "describe", "--tags", "--exact-match"], {
      encoding: "utf8",
      windowsHide: true,
    });

    if (result.status === 0) {
      const tag = result.stdout.trim();
      if (tag) {
        return tag;
      }
    }
  }

  return undefined;
}

function repositoryName(root = packageRoot()) {
  const fromEnv = process.env.XD_KB_GITHUB_REPOSITORY || process.env.GITHUB_REPOSITORY;
  if (fromEnv) {
    return fromEnv.replace(/^https:\/\/github\.com\//, "").replace(/\.git$/, "");
  }

  const pkg = readPackageJson(root);
  const repo = typeof pkg.repository === "string" ? pkg.repository : pkg.repository?.url;
  const match = repo && repo.match(/github\.com[:/](.+?)(?:\.git)?$/);
  return match ? match[1] : "zhing2006/xd-kb";
}

function releaseUrl(repo, tag, file) {
  return `https://github.com/${repo}/releases/download/${encodeURIComponent(tag)}/${encodeURIComponent(file)}`;
}

function existsAndNonEmpty(file) {
  return fs.existsSync(file) && fs.statSync(file).size > 0;
}

function githubHeaders(accept = "application/vnd.github+json") {
  const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN;
  const headers = {
    "user-agent": "xd-kb-npm-pack",
    accept,
  };

  if (token) {
    headers.authorization = `Bearer ${token}`;
  }

  return headers;
}

async function fetchJson(url) {
  const response = await fetch(url, {
    headers: githubHeaders("application/vnd.github+json"),
    redirect: "follow",
  });

  if (!response.ok) {
    const extra =
      response.status === 404
        ? " If this is a private repository, set GITHUB_TOKEN or GH_TOKEN."
        : "";
    throw new Error(`GET ${url} failed with ${response.status} ${response.statusText}.${extra}`);
  }

  return response.json();
}

async function resolveReleaseAssets(repo, tag) {
  const url = `https://api.github.com/repos/${repo}/releases/tags/${encodeURIComponent(tag)}`;
  const release = await fetchJson(url);
  const assets = new Map();

  for (const asset of release.assets || []) {
    assets.set(asset.name, asset);
  }

  return assets;
}

function ghCommand() {
  return process.env.GH_PATH || "gh";
}

function ghAvailable() {
  const result = spawnSync(ghCommand(), ["--version"], {
    stdio: "ignore",
    windowsHide: true,
  });

  return result.status === 0;
}

function downloadWithGh(repo, tag, vendorDir, assets, force) {
  const args = ["release", "download", tag, "--repo", repo, "--dir", vendorDir];

  for (const asset of assets) {
    args.push("--pattern", asset.file);
  }

  if (force) {
    args.push("--clobber");
  }

  const result = spawnSync(ghCommand(), args, {
    stdio: "inherit",
    windowsHide: false,
  });

  return !result.error && result.status === 0;
}

async function downloadToFile(url, destination, accept = "application/octet-stream") {
  const headers = githubHeaders(accept);
  const response = await fetch(url, { headers, redirect: "follow" });
  if (!response.ok) {
    throw new Error(`GET ${url} failed with ${response.status} ${response.statusText}`);
  }

  const body = Buffer.from(await response.arrayBuffer());
  if (body.length === 0) {
    throw new Error(`GET ${url} returned an empty file`);
  }

  const temporary = `${destination}.download`;
  await fs.promises.writeFile(temporary, body);
  await fs.promises.rename(temporary, destination);
}

async function ensureReleaseAssets(options = {}) {
  const root = options.packageRoot || packageRoot();
  const tag = options.tag || process.env.XD_KB_RELEASE_TAG || defaultTag(root);
  const repo = options.repo || repositoryName(root);
  const force = Boolean(options.force);
  const vendorDir = path.join(root, "vendor");
  // Sibling of vendor/, NOT inside it — package.json `files: ["vendor", ...]`
  // would otherwise pack the marker into the npm tarball.
  const tagMarker = path.join(root, ".vendor-tag");

  await fs.promises.mkdir(vendorDir, { recursive: true });

  // Detect a stale vendor/: if the marker is missing or names a different
  // tag than the one we are about to package, treat the whole vendor as
  // out-of-date so we never repeat the v0.1.0-binaries-in-v0.1.1-tarball
  // mistake. This effectively promotes the implicit cache hit to an
  // explicit `--force`.
  const previousTag = readVendorTag(tagMarker);
  let effectiveForce = force;
  if (!effectiveForce && previousTag !== tag) {
    if (previousTag) {
      console.log(`vendor cache is for ${previousTag}; refetching for ${tag}`);
    }
    effectiveForce = true;
  }

  const pending = [];

  for (const asset of ASSETS) {
    const destination = path.join(vendorDir, asset.file);
    const exists = existsAndNonEmpty(destination);

    if (exists && !effectiveForce) {
      if (asset.executable) {
        await fs.promises.chmod(destination, 0o755).catch(() => undefined);
      }
      console.log(`skip ${asset.file}`);
      continue;
    }

    pending.push(asset);
  }

  if (pending.length === 0) {
    // Either every asset was skipped (cache hit) or there were no assets
    // to fetch in the first place. Either way the vendor matches `tag`,
    // so refresh the marker to record that fact.
    writeVendorTag(tagMarker, tag);
    return;
  }

  if (process.env.XD_KB_NO_GH !== "1" && ghAvailable()) {
    console.log(`download release assets with gh from ${repo}@${tag}`);
    // Pass `effectiveForce` (not raw `force`) so a stale-cache refetch
    // gets `--clobber` even when the user didn't pass --force themselves.
    if (downloadWithGh(repo, tag, vendorDir, pending, effectiveForce)) {
      const missing = pending.filter((asset) => !existsAndNonEmpty(path.join(vendorDir, asset.file)));
      if (missing.length === 0) {
        for (const asset of pending) {
          if (asset.executable) {
            await fs.promises.chmod(path.join(vendorDir, asset.file), 0o755).catch(() => undefined);
          }
        }
        writeVendorTag(tagMarker, tag);
        return;
      }

      console.warn(`gh did not download: ${missing.map((asset) => asset.file).join(", ")}`);
    } else {
      console.warn("gh release download failed; falling back to GitHub API");
    }
  }

  const releaseAssets = await resolveReleaseAssets(repo, tag);

  for (const asset of pending) {
    const destination = path.join(vendorDir, asset.file);
    const releaseAsset = releaseAssets.get(asset.file);
    if (!releaseAsset) {
      const available = [...releaseAssets.keys()].sort();
      const suffix = available.length ? ` Available assets: ${available.join(", ")}` : "";
      throw new Error(`Release ${repo}@${tag} does not contain ${asset.file}.${suffix}`);
    }

    const url = releaseAsset.url || releaseAsset.browser_download_url || releaseUrl(repo, tag, asset.file);
    console.log(`download ${asset.file} from ${repo}@${tag}`);
    await downloadToFile(url, destination);

    if (asset.executable) {
      await fs.promises.chmod(destination, 0o755).catch(() => undefined);
    }
  }

  // All assets fetched successfully — record the tag so the next pack
  // for the same tag short-circuits and the next pack for a different
  // tag triggers an automatic refetch.
  writeVendorTag(tagMarker, tag);
}

function readVendorTag(file) {
  try {
    return fs.readFileSync(file, "utf8").trim() || undefined;
  } catch (error) {
    if (error.code === "ENOENT") {
      return undefined;
    }
    throw error;
  }
}

function writeVendorTag(file, tag) {
  fs.writeFileSync(file, `${tag}\n`);
}

module.exports = {
  ASSETS,
  currentGitTag,
  defaultTag,
  ensureReleaseAssets,
  packageRoot,
  repositoryName,
};
