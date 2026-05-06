#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { ASSETS } = require("./release-assets");

const vendorDir = path.join(__dirname, "..", "vendor");

for (const asset of ASSETS) {
  if (!asset.executable) {
    continue;
  }

  const file = path.join(vendorDir, asset.file);
  if (!fs.existsSync(file)) {
    continue;
  }

  try {
    fs.chmodSync(file, 0o755);
  } catch {
    // Do not fail installation; the command shim reports runtime errors clearly.
  }
}
