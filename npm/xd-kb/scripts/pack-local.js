#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { defaultTag, ensureReleaseAssets, packageRoot } = require("./release-assets");

function npmInvocation() {
  const candidates = [];

  if (process.env.npm_execpath) {
    candidates.push(process.env.npm_execpath);
  }

  candidates.push(path.join(path.dirname(process.execPath), "node_modules", "npm", "bin", "npm-cli.js"));

  for (const candidate of candidates) {
    if (candidate.endsWith(".js") && fs.existsSync(candidate)) {
      return { command: process.execPath, args: [candidate] };
    }
  }

  return { command: process.platform === "win32" ? "npm.cmd" : "npm", args: [] };
}

async function main() {
  const root = packageRoot();
  const args = process.argv.slice(2);
  const force = args.includes("--force");
  const tagArg = args.find((arg) => !arg.startsWith("-"));
  const tag = process.env.XD_KB_RELEASE_TAG || tagArg || defaultTag(root);
  const destination =
    process.env.XD_KB_PACK_DEST || path.resolve(root, "..", "..", "..", "packages");

  await ensureReleaseAssets({ packageRoot: root, tag, force });
  await fs.promises.mkdir(destination, { recursive: true });

  const npm = npmInvocation();
  const result = spawnSync(npm.command, [...npm.args, "pack", root, "--pack-destination", destination], {
    stdio: "inherit",
    windowsHide: false,
  });

  if (result.error) {
    throw result.error;
  }

  process.exit(result.status ?? 1);
}

main().catch((error) => {
  console.error(`xd-kb: failed to create local npm package: ${error.message}`);
  process.exit(1);
});
