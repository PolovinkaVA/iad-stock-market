import { FileBlob, PresentationFile } from "file:///C:/Users/skyda/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const path = "reports/Презентация_к_защите_SBER_проверенная.pptx";
const presentation = await PresentationFile.importPptx(await FileBlob.load(path));
const inspection = await presentation.inspect({ kind: "slide,textbox,table,chart,notes", maxChars: 40000 });
for (const line of inspection.ndjson.split("\n")) {
  if (!line.trim()) continue;
  const item = JSON.parse(line);
  if ([3].includes(item.slideIndex)) console.log(line);
}
