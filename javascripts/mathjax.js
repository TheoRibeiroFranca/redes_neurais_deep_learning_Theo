window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"],
    // só processa o que o arithmatex marcou como fórmula
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset after each page load (Material SPA)
// Na navegação instantânea o MathJax fica preso a uma folha de estilos que saiu do
// documento ("can't insert css rule") e as fórmulas novas aparecem em branco. Antes
// de cada re-typeset, a folha velha é descartada para ele criar outra. Na primeira
// carga não há nada a fazer: o próprio MathJax faz o typeset.
let primeiraCarga = true;
document$.subscribe(() => {
  if (primeiraCarga) {
    primeiraCarga = false;
    return;
  }
  const output = window.MathJax?.startup?.output;
  if (!output) return;
  document.getElementById("MJX-CHTML-styles")?.remove();
  output.chtmlStyles = null;
  output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
