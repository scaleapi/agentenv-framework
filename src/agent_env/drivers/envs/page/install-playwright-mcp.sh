#!/bin/sh
# Node and Playwright MCP, from the tarballs the install downloaded and checked, with no npm install, since an install
# has no network: Node's binary and licence in node/, and each npm package in node_modules/ where npm would put it, by
# the name its package.json gives. A package whose dependencies aren't all there, at the versions it names, fails it.
set -eu
work=$(mktemp -d)
tar -xzf node.tar.gz -C "$work"
mkdir -p node/bin
mv "$work"/node-*/bin/node node/bin/node
mv "$work"/node-*/LICENSE node/LICENSE
rm -rf "$work" node.tar.gz
for package in *.tgz; do
    work=$(mktemp -d)
    # mktemp's folder is its owner's alone, and the session user runs what's in it.
    chmod 755 "$work"
    tar -xzf "$package" -C "$work" --strip-components=1
    name=$(node/bin/node -p 'require(process.argv[1]).name' "$work/package.json")
    mkdir -p "node_modules/$(dirname "$name")"
    mv "$work" "node_modules/$name"
    rm -f "$package"
done
node/bin/node -e '
const fs = require("fs"), path = require("path");
const names = fs.readdirSync("node_modules").flatMap(n => n.startsWith("@")
    ? fs.readdirSync(path.join("node_modules", n)).map(m => `${n}/${m}`) : [n]);
const unmet = names.flatMap(name => Object.entries(
    require(path.resolve("node_modules", name, "package.json")).dependencies || {}).flatMap(([dependency, version]) => {
    const file = path.resolve("node_modules", dependency, "package.json");
    const found = fs.existsSync(file) ? require(file).version : "none";
    return found === version ? [] : [`${name} needs ${dependency} ${version}, found ${found}`];
}));
if (unmet.length) { console.error(unmet.join("\n")); process.exit(1); }
'
