#import "style.typ": *
#import "portada.typ": portada

// ============================================
// VARIABLES DEL DOCUMENTO
// ============================================
#let titulo = "Título del trabajo"
#let autor = "Emilio Manuel Vázquez Cruz"
#let tutor = "Juan Galán Paez"
#let grado = "Grado en Ingeniería Informática - Ingeniería del Software"
#let universidad = "Universidad de Sevilla"
#let departamento = "Ciencias de la Computación e Inteligencia Artificial"
#let convocatoria = "junio"
#let curso = "2025/26"
#let dedicatoria = "Aquí la dedicatoria del trabajo"

// ============================================
// PORTADA Y SECCIONES PRELIMINARES
// ============================================
#portada(titulo, autor, tutor, grado, universidad, departamento, convocatoria, curso)

#include "sections/00_agradecimientos.typ"
#include "sections/00_resumen.typ"
#include "sections/00_abstract.typ"

// ============================================
// ÍNDICES
// ============================================
#show outline.entry.where(level: 1): set text(weight: "bold")

#outline(
  title: "Índice",
  depth: 2,
  indent: 1em,
)

#outline(
  title: "Índice de figuras",
  target: figure.where(kind: image),
)

#outline(
  title: "Índice de tablas",
  target: figure.where(kind: table),
)

#outline(
  title: "Índice de extractos de código",
  target: figure.where(kind: raw),
)

#counter(page).update(1)
#set page(numbering: "1")

// ============================================
// CAPÍTULOS
// ============================================
#include "sections/01_Introduccion.typ"
#include "sections/02_Gestion.typ"
#include "sections/03_Analisis.typ"
#include "sections/04_Diseño.typ"
#include "sections/05_Implementacion.typ"
#include "sections/06_Pruebas.typ"
#include "sections/XX_Conclusiones.typ"

// ============================================
// BIBLIOGRAFÍA
// ============================================
#bibliography("bibliografia.bib", style: "ieee", title: "Bibliografía")