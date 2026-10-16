// Rasterize generated SVG figures for Word. Requires the sharp Node package.
const fs = require('node:fs');
const path = require('node:path');
const sharp = require('sharp');
const directory = path.resolve(__dirname, '../reports/figures');
async function main() {
  for (const name of fs.readdirSync(directory).filter(name => name.endsWith('.svg'))) {
    await sharp(path.join(directory, name), {density: 180})
      .png().toFile(path.join(directory, name.replace(/\.svg$/, '.png')));
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
