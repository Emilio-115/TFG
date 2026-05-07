#import "@preview/deal-us-tfc-template:1.0.0": *

= Introducción
<sec:introduccion>

En este capítulo se establece el marco general del presente Trabajo de Fin de Grado. Se comienza contextualizando el impacto de las patologías del suelo pélvico en la salud pública, seguido de una exposición de los desafíos actuales en su diagnóstico mediante ecografía. Finalmente, se detallan la motivación y los objetivos que articulan este trabajo, centrada en el desarrollo de modelos predictivos para el diagnóstico.

== Contexto clínico y motivación

La *uroginecología* se enfrenta hoy a un reto creciente, la alta prevalencia de las *Disfunciones del Suelo Pélvico*, entre las que destacan los diversos tipos de *prolapsos de órganos pélvicos (POP)* como el cistocele, el rectocele o el prolapso uterino. Estas condiciones no solo afectan la anatomía funcional de la paciente, sino que deterioran gravemente su calidad de vida.

Actualmente, el diagnóstico clínico se apoya principalmente en métodos manuales. Algunos de estos métodos pueden ser, el uso de sistemas de clasificación como *POP-Q* para medir el grado del prolapso, *pruebas funcionales* en las que se estudia el *comportamiento* de los órganos, técnicas de imagen como es la *ecografía transperineal dinámica*. Sin embargo, esta técnica presenta una limitación crítica, su *alta dependencia del operador*. La interpretación de los fotogramas, la identificación de las estructuras (como el músculo elevador del ano o la uretra) o la medición en diferentes prácticas requieren una curva de aprendizaje extensa y están sujetas a una presente variabilidad subjetiva.

En los últimos años se ha propuesto el uso de sistemas impulsados por inteligencia artificial para apoyar y acelerar evaluaciones en el ámbito de la medicina y, concretamente, se incluye el caso para prolapsos de órganos pélvicos. 

La combinación de una problemática clínica relevante, como es la variabilidad en el diagnóstico del prolapso de órganos pélvicos, y la existencia de herramientas emergentes como la inteligencia artificial, plantea un escenario especialmente atractivo desde el punto de vista ingenieril. Este contexto no solo evidencia una necesidad real de mejora en los métodos diagnósticos actuales, sino que también abre la puerta a la aplicación de técnicas avanzadas de análisis de datos y visión por computador. En este sentido, el problema deja de ser exclusivamente médico para convertirse en un *desafío interdisciplinar*, donde la ingeniería puede aportar soluciones concretas. Es precisamente en este punto donde surge la motivación de este trabajo, al identificar una oportunidad clara de aplicar conocimientos técnicos en un ámbito con impacto directo en la práctica clínica y en la calidad de vida de las pacientes.

Inicialmente, mi interés se orientaba hacia aprender sobre el desarrollo de agentes de aprendizaje por refuerzo aplicados al ámbito de los videojuegos, un entorno con el que estoy familiarizado. Sin embargo, tras un proceso de reflexión sobre el propósito de mi formación como ingeniero, decidí reorientar mi esfuerzo hacia un problema con una aplicación real en el sector de la salud.


La propuesta de aplicar inteligencia artificial para la detección de prolapsos en el suelo pélvico a partir de fotogramas de ecografías captó mi atención de inmediato. La posibilidad de aportar una contribución a la aplicación de inteligencia artificial para el diagnóstico de estas patologías representaba una oportunidad única para dar un propósito práctico y humano a las habilidades adquiridas durante el grado.

Soy consciente de que, como estudiante de ingeniería informática, mis conocimientos iniciales en medicina y, específicamente, en uroginecología, son simplemente muy superficiales. No obstante, considero que esta brecha de conocimiento ha sido uno de los mayores alicientes del proyecto. La necesidad de sumergirme en un campo externo me ha permitido obtener:

- *Interdisciplinariedad*: Aprender sobre un campo desconocido para mí y colaborar con las necesidades de un especialista clínico.
- *Adaptabilidad técnica*: Aplicar metodologías de análisis de datos e ingeniería de variables en un entorno donde el "ruido" y la variabilidad biológica suponen un desafío mucho mayor que en los entornos digitales controlados.
- *Aprendizaje de nuevas tecnologías*: Implementar arquitecturas avanzadas para el entrenamiento de modelos de inteligencia artificial. Así como trabajar con datos provenientes de fotogramas.

Para este trabajo se disponen de datos los fotogramas con los órganos segmentados para 211 videos de ecografías de pacientes sanos y de pacientes que presentas prolapsos del suelo pélvico entre los que se incluyen cistocele,cystourethrocele, prolapso uterino, elongación cervical, rectocele y enterocele.

== Objetivos


