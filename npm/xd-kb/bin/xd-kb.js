#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const TARGETS = {
  "win32:x64": "xd-kb-win32-x64.exe",
  "linux:x64": "xd-kb-linux-x64",
  "linux:arm64": "xd-kb-linux-arm64",
  "darwin:x64": "xd-kb-darwin-x64",
  "darwin:arm64": "xd-kb-darwin-arm64",
};

const key = `${process.platform}:${process.arch}`;
const asset = TARGETS[key];

if (!asset) {
  console.error(
    `xd-kb: unsupported platform ${process.platform}/${process.arch}. ` +
      `Supported platforms: ${Object.keys(TARGETS).join(", ")}.`
  );
  process.exit(1);
}

const binary = path.join(__dirname, "..", "vendor", asset);

if (!fs.existsSync(binary)) {
  console.error(
    `xd-kb: missing native binary for ${process.platform}/${process.arch}.\n` +
      `Expected: ${binary}\n` +
      `Rebuild the npm package with: npm run pack:local`
  );
  process.exit(1);
}

if (process.platform !== "win32") {
  try {
    fs.chmodSync(binary, 0o755);
  } catch {
    // Best-effort: the spawn error below will report a real permission failure.
  }
}

const result = spawnSync(binary, process.argv.slice(2), {
  stdio: "inherit",
  windowsHide: false,
});

if (result.error) {
  console.error(`xd-kb: failed to start native binary: ${result.error.message}`);
  process.exit(1);
}

if (result.signal) {
  process.kill(process.pid, result.signal);
}

process.exit(result.status ?? 1);
