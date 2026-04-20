En esta carpeta se encuentran todos aquellos experimentos realizados hasta el commit 660b135, en el se tiene un for en el que se realizan múltiples ejecuciones del experimento, con su Stratified y en cada iteracion la particion para entrenar y evaluar es aleatoria.

Se ha probado con la columna case, nhc y frames en el dataset, sin ellas y con el scale_pos_weight para potenciar aquellos entrenos con menor número de casos positivos.

Se ha probado a predecir a nivel de fotograma individual y agrupando para los diferentes casos y tomando para la predicción la media. Predecir a nivel de fotograma no tenía mucho sentido dejarlo como tal pues al final la enfermedad va por caso en cuestión por lo que los resultados no representarían completamente la realidad.
