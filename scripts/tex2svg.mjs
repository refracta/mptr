#!/usr/bin/env node
import fs from "node:fs";

import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";

function readStdin() {
  return fs.readFileSync(0, "utf-8");
}

function findFirstSvg(adaptor, node) {
  if (!node) return null;
  const kind = adaptor.kind(node);
  if (typeof kind === "string" && kind.toLowerCase() === "svg") return node;
  let child = adaptor.firstChild(node);
  while (child) {
    const found = findFirstSvg(adaptor, child);
    if (found) return found;
    child = adaptor.nextSibling(child);
  }
  return null;
}

function main() {
  const input = readStdin().trim();
  const parsed = input ? JSON.parse(input) : [];
  const reqs = Array.isArray(parsed) ? parsed : [parsed];

  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);

  const tex = new TeX({ packages: AllPackages });
  const svg = new SVG({ fontCache: "none" });
  const html = mathjax.document("", { InputJax: tex, OutputJax: svg });

  const out = reqs.map((req) => {
    const texString = String(req?.tex ?? "");
    const display = Boolean(req?.display);
    try {
      const node = html.convert(texString, { display });
      const svgNode = findFirstSvg(adaptor, node) ?? node;
      return { tex: texString, display, svg: adaptor.outerHTML(svgNode) };
    } catch (e) {
      return { tex: texString, display, svg: null, error: String(e?.message ?? e) };
    }
  });

  process.stdout.write(JSON.stringify(out));
}

main();

