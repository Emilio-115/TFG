#let portada(
  titulo,
  autor,
  tutor,
  grado,
  universidad,
  departamento,
  convocatoria,
  curso,
  dedicatoria: none
) = [
  #set page(margin: 2.5cm, numbering: none)
  #align(center)[
    #v(1cm)
    #image("etsii_us.png", width: 100%)
    #v(3cm)

    #text(size: 14pt)[TRABAJO FIN DE GRADO]
    #v(0.1in)
    #text(size: 28pt, weight: "bold")[#titulo]
    #v(0.2in)

    #text(size: 14pt)[Realizado por]
    #linebreak()
    #text(size: 18pt, weight: "bold")[#autor]
    #v(3cm)

    #text(weight: "bold")[Para la obtención del título de]
    #linebreak()
    #text(size: 14pt)[#grado]
    #v(0.2in)

    #text(weight: "bold")[Dirigido por]
    #linebreak()
    #text(size: 14pt)[#tutor]
    #v(0.2in)

    #text(weight: "bold")[En el departamento de]
    #linebreak()
    #text(size: 14pt)[#departamento]
    #v(0.6in)

    #text(size: 18pt, weight: "bold")[Convocatoria de #convocatoria, curso #curso]
  ]

  #if dedicatoria != none [
    #pagebreak()
    #set page(numbering: none)
    #v(1fr)
    #align(center)[
      #emph[#dedicatoria]
    ]
    #v(1fr)
  ]

  #pagebreak()
  #counter(page).update(1)
  #set page(numbering: "i")
]