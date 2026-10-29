// Test harness for the HTML viewer's inline script (used by test_longecho_site.py).
//
// Usage: node viewer_harness.js <inline-script.js> [<site-dir>] < program.js
//
// Runs the page's inline script in a vm context with a minimal DOM stub (the
// boot calls at the end are cut off), after the site's sql-wasm.js and
// sql-wasm-binary.js when a site dir is given. Then runs the program from
// stdin as an async function body with `ctx` (the page's global scope) in
// scope, and prints its return value as JSON. <script src> tags appended to
// document.head are emulated: the file is read from the site dir and run in
// the same context, then onload fires; a missing file fires onerror.
"use strict";
const vm = require("vm");
const fs = require("fs");
const path = require("path");

const [scriptPath, siteDir] = process.argv.slice(2);

function element(id) {
  return {
    id, textContent: "", style: {}, children: [], firstChild: null,
    classList: { add() {}, remove() {} },
    addEventListener() {},
    insertBefore(node) { this.children.unshift(node); this.firstChild = node; },
  };
}

const elements = {};
let ctx;
const head = {
  removeChild() {},
  appendChild(tag) {
    tag.parentNode = head;
    setImmediate(() => {
      const file = path.join(siteDir || "", tag.src);
      if (!siteDir || !fs.existsSync(file)) return tag.onerror();
      try {
        vm.runInContext(fs.readFileSync(file, "utf8"), ctx);
      } catch (e) {
        // A browser reports a throwing script on window and still fires load.
        process.stderr.write("script error in " + tag.src + ": " + e.message + "\n");
      }
      tag.onload();
    });
  },
};

const document = {
  head,
  getElementById(id) { return elements[id] || (elements[id] = element(id)); },
  createElement(tag) {
    if (tag !== "div") return element(null);
    // esc() relies on textContent -> innerHTML escaping.
    const div = { textContent: "" };
    Object.defineProperty(div, "innerHTML", {
      get() {
        return String(this.textContent)
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      },
    });
    return div;
  },
};

ctx = vm.createContext({
  document, atob, Blob, Response, DecompressionStream, TextDecoder, TextEncoder,
  WebAssembly, URLSearchParams, setTimeout,
  console: new console.Console(process.stderr),  // keep stdout for the result
  location: { search: "" },
});
ctx.window = ctx;

if (siteDir) {
  for (const name of ["sql-wasm.js", "sql-wasm-binary.js"]) {
    const file = path.join(siteDir, name);
    if (fs.existsSync(file)) vm.runInContext(fs.readFileSync(file, "utf8"), ctx);
  }
}

const page = fs.readFileSync(scriptPath, "utf8");
const boot = page.lastIndexOf("initTheme();");
if (boot < 0) throw new Error("boot marker initTheme(); not found");
vm.runInContext(page.slice(0, boot), ctx);

const AsyncFunction = (async function () {}).constructor;
const program = new AsyncFunction("ctx", fs.readFileSync(0, "utf8"));
program(ctx).then(
  (result) => process.stdout.write(JSON.stringify(result === undefined ? null : result)),
  (err) => { process.stderr.write(String(err && err.stack || err) + "\n"); process.exit(1); },
);
