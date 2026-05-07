#import "@preview/deal-us-tfc-template:1.0.0": *

#show: TFC.with(
  titulo: "Trabajo fin de grado",
  alumno: "Emilio Manuel Vázquez Cruz",
  titulacion: "Grado en Ingeniería Informática - Ingeniería del Software",
  director: [Juan Galán Páez],
  departamento: "Ciencias de la Computación e Inteligencia Artificial",
  convocatoria: "Convocatoria de junio, curso 2025/26",
  dedicatoria: "Aquí la dedicatoria del trabajo",
  agradecimientos: [
    Quiero agradecer a X por...

    También quiero agradecer a Y por...
  ],
  resumen: [
    Incluya aquí un resumen de los aspectos generales de su trabajo, en español
  ],
  palabras-clave: (
    "palabra clave 1", 
    "palabra clave 2", 
    "...", 
    "palabra clave N"
  ),
  abstract: [
    This section should contain an English version of the Spanish abstract.
  ],
  keywords: (
    "keyword 1", 
    "keyword 2", 
    "...", 
    "keyword N"
  )
)

#include "sections/ejemplos_borrame.typ"
#include "sections/01_introduccion.typ"
#include "sections/02_fundamentos.typ"
#include "sections/03_datos_ing_variables.typ"
#include "sections/04_metodologia.typ"
#include "sections/05_experimentacion_resultados.typ"
#include "sections/06_Pruebas.typ"
#include "sections/XX_Conclusiones.typ"

#bibliography("bibliografia.bib")