#!/usr/bin/env node

const { defaultTag, ensureReleaseAssets, packageRoot } = require("./release-assets");

const args = process.argv.slice(2);
const force = args.includes("--force");
const tagArg = args.find((arg) => !arg.startsWith("-"));
const root = packageRoot();
const tag = process.env.XD_KB_RELEASE_TAG || tagArg || defaultTag(root);

ensureReleaseAssets({ packageRoot: root, tag, force }).catch((error) => {
  console.error(`xd-kb: failed to fetch release assets: ${error.message}`);
  process.exit(1);
});
