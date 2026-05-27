# Ingeniería de características y modelos de series temporales para el diagnóstico del prolapso de suelo pélvico: un estudio comparativo

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado
del Grado en Ingeniería Informática. El trabajo aborda el diagnóstico automático
de prolapsos de órganos pélvicos (POP) a partir de series temporales
extraídas de vídeos de ecografía, combinando ingeniería de
características y modelos de aprendizaje automático.

Se implementan y comparan cuatro modelos: XGBoost, ResNet1D, TCN y BiLSTM.
El flujo incluye preprocesamiento mediante ventana deslizante, selección de variables
por importancia por permutación y optimización bayesiana de hiperparámetros con
validación cruzada anidada estratificada por grupos.
