#!/bin/sh
# Node and Playwright MCP, from the tarballs the install downloaded and checked, with no npm install, since an install
# has no network: Node's binary and licence in node/, and Playwright MCP's three npm packages in node_modules/, each
# where npm would put it.
set -eu
work=$(mktemp -d)
tar -xzf node.tar.gz -C "$work"
mkdir -p node/bin
mv "$work"/node-*/bin/node node/bin/node
mv "$work"/node-*/LICENSE node/LICENSE
rm -rf "$work" node.tar.gz
unpack() {
    mkdir -p "node_modules/$2"
    tar -xzf "$1" -C "node_modules/$2" --strip-components=1
    rm -f "$1"
}
unpack playwright-mcp.tgz @playwright/mcp
unpack playwright.tgz playwright
unpack playwright-core.tgz playwright-core
