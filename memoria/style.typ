// ============================================
// HEADINGS (equivalente a \titleformat)
// ============================================

#show heading.where(level: 1): it => {
  // Equivalente a \chapter
  set text(size: 24pt, weight: "bold")
  block(above: 25pt, below: 12pt)[
    #it.body
    #v(2pt)
    #line(length: 100%, stroke: 2pt)
  ]
}

#show heading.where(level: 2): it => {
  // Equivalente a \section
  set text(size: 16pt, weight: "bold")
  block(above: 14pt, below: 8pt, it)
}


// ============================================
// ALIAS DE TEXTO (equivalente a \newcommand)
// ============================================

#let negritas(body) = strong(body)
#let cursiva(body) = emph(body)
#let codigo(body) = raw(body)


// ============================================
// PÁGINA Y MÁRGENES (equivalente a \geometry)
// ============================================

#set page(
  paper: "a4",
  margin: 2.75cm,
)


// ============================================
// PÁRRAFOS
// ============================================

#set par(
  first-line-indent: 0.75cm,
  justify: true,
)


// ============================================
// CÓDIGO FUENTE (equivalente a lstset)
// ============================================

#show raw.where(block: true): it => {
  set text(font: "DejaVu Sans Mono", size: 9pt)
  block(
    fill: white,
    stroke: (top: 1pt, bottom: 1pt),
    inset: (x: 8pt, y: 6pt),
    width: 100%,
    it
  )
}

// Colores para syntax highlighting (si usas un tema personalizado)
#let code-green = rgb(0, 124, 0)
#let code-gray = rgb(128, 128, 128)
#let code-purple = rgb(148, 0, 209)
#let code-ocher = rgb(204, 77, 0)


// ============================================
// TOC (equivalente a \DeclareTOCStyleEntry)
// ============================================

#show outline.entry.where(level: 1): it => {
  set text(size: 13pt, weight: "bold")
  block(above: 12pt, it)
}

#show outline.entry.where(level: 2): it => {
  set text(size: 11pt)
  it
}
