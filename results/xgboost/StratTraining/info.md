En esta sección se almacenan los resultados del nuevo metodo main. El bucle for y el entrnamiento se ha modificado. Ahora en vez de hacer una division, paso seguido entrenar y, con lo resultante, evaluar, aquí se recorren los diferentes folds estratificados, se optimizan los parametros como antes, se obtiene un modelo con los parametros óptimizados y se evalúa sobre el fold restante, este proceso se repite para las otras divisiones.

Se ha probado a modificar el tamaño de la ventana, por defecto es 60 con paso 30, he probado 30 con paso 15, 90 con paso 30.

A partir de la numeración 2\_ en los resultados aparecen las metricas average precission y roc_auc.

Se ha probado a añadir la variable distancia entre todos los órganos, y la excentricidad. A través de permutation importance se ha obtenido cuáles son las más importantes.

Se han probado tres variantes para ver el impacto, 1. se han dejado las 80 mejores de la lista, 2. Se han juntando las originales junto a las 10 mejores de la lista y 3. Las originales más la excentricidad.

Añadir las 10 variables distancia ayuda a obtener mejores resultados para un par de enfermedades. Empeora ligeramente el resto.
PROBAR CON ESTE NUEVO MÉTODO DE LAS 10 MEJORES LA IMPORTANCIA DE VARIABLES.



He probado a hacer un ranking global mejor general. Probar cosas más concretas como añadir variables de los organos más flojos. Como la cervical elong, rectocele y enterocele.


En general, usar las variables con más efecto para cada organo no siempre ha resultado mejor.pco


Probar a dejar las distancias a las que afecta cada organo



SINCRONIZAR LO QUE HAY EN JUPYTER CON LOCAL






