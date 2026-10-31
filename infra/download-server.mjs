// Fetch the vanilla server jar for a version, checksum it, write it out.
// Same shape as mine-ai-mcp's docker/download-server.mjs, but with an output path.
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

const version = process.argv[2] ?? '1.21.4';
const out = process.argv[3] ?? '/opt/server.jar';

const manifest = await (await get('https://piston-meta.mojang.com/mc/game/version_manifest_v2.json')).json();
const entry = manifest.versions.find(item => item.id === version);
if (!entry) throw new Error(`Unknown Minecraft version: ${version}`);

const metadata = await (await get(entry.url)).json();
const server = metadata.downloads.server;
const bytes = Buffer.from(await (await get(server.url)).arrayBuffer());
if (createHash('sha1').update(bytes).digest('hex') !== server.sha1) {
  throw new Error('Minecraft server checksum mismatch');
}

await writeFile(out, bytes);
console.log(`server.jar ${version} -> ${out} (${bytes.length} bytes)`);
