gh repo clone romera-rocio/Titanic_Rocio_Romera

# Titanic ML Analysis – Advanced Data Science Portfolio Project

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)]
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)]
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2-green?logo=scikit-learn)]
[![Pandas](https://img.shields.io/badge/Pandas-1.6-blue?logo=pandas)]
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13-orange?logo=seaborn)]
[![License](https://img.shields.io/badge/License-MIT-green)]

---

## 1. Introducción

El hundimiento del RMS Titanic en 1912 constituye un caso paradigmático para aplicar **estadística avanzada**, **análisis exploratorio de datos (EDA)**, **ingeniería de características**, **preprocesamiento de datos estructurados** y **modelos supervisados**.

Este proyecto analiza el dataset de [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) con un enfoque profesional, reproducible y escalable, incluyendo contrastes estadísticos, pipelines de preprocesamiento y modelado.

---

## 2. Objetivos

**Objetivo general:**
Analizar sistemáticamente los factores determinantes de la supervivencia y construir un flujo de trabajo reproducible en ML.

**Objetivos específicos:**

* Formular y contrastar hipótesis estadísticas (H0/H1).
* Realizar EDA univariado, bivariado y multivariado.
* Construir variables derivadas avanzadas (`FamilySize`, `IsAlone`, `Title`, `Deck`).
* Detectar y manejar valores faltantes y outliers.
* Analizar correlaciones según tipo de variable (numérica, categórica).
* Implementar pipelines reproducibles: imputación, escalado, codificación.
* Entrenar modelos supervisados: Decision Tree, KNN y Random Forest.
* Evaluar modelos con métricas robustas: Accuracy, ROC-AUC, Confusion Matrix.
* Generar predicciones listas para envío a Kaggle.

---

## 3. Hipótesis estadísticas (H0/H1)

| H   | Variable / Interacción       | Hipótesis Nula (H0)                 | Hipótesis Alternativa (H1) |
| --- | ----------------------------- | ------------------------------------ | --------------------------- |
| H1  | Sexo                          | No afecta supervivencia              | Afecta supervivencia        |
| H2  | Clase (Pclass)                | No influye                           | Sí influye                 |
| H3  | Edad                          | No está asociada                    | Sí está asociada          |
| H4  | Tarifa (Fare)                 | No afecta                            | Sí afecta                  |
| H5  | Puerto de embarque (Embarked) | No influye                           | Sí influye                 |
| H6  | Tamaño familiar (FamilySize) | No está asociada                    | Sí está asociada          |
| H7  | Cabina (Cabin/Deck)           | No afecta                            | Sí afecta                  |
| H8  | Título social (Title)        | No aporta información               | Sí aporta información     |
| H9  | Sexo × Clase                 | No mejora explicación               | Sí mejora explicación     |
| H10 | Ticket prefix                 | No asociado                          | Sí asociado                |
| H11 | Fare × Pclass                | No afecta                            | Sí afecta                  |
| H12 | Modelos no lineales           | No mejoran                           | Sí mejoran                 |
| H13 | Estandarización              | No mejora modelos sensibles a escala | Sí mejora                  |
| H14 | Imputación avanzada de Age   | No mejora predicción                | Sí mejora predicción      |
| H15 | Features derivadas            | No incrementan rendimiento           | Sí incrementan rendimiento |

---

## 4. Metodología

### 4.1 Librerías

* **Ciencia de datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **ML y pipelines:** `scikit-learn`
* **Estadística:** `scipy.stats`

### 4.2 Flujo analítico

1. Carga de datos (`train.csv`, `test.csv`)
2. EDA univariado, bivariado y multivariado
3. Detección de valores faltantes, outliers y análisis de distribución
4. Ingeniería de variables derivadas (`FamilySize`, `IsAlone`, `Title`, `Deck`)
5. Preprocesamiento reproducible mediante pipelines
6. Entrenamiento y evaluación de modelos supervisados
7. Validación cruzada y métricas robustas (Accuracy, ROC-AUC, Confusion Matrix)
8. Predicciones sobre dataset de test

---

## 5. EDA – Estadística descriptiva y exploratoria

### 5.1 Estadísticas resumidas

| Variable | Media | Mediana | Min  | Max | Observaciones                                |
| -------- | ----- | ------- | ---- | --- | -------------------------------------------- |
| Age      | 29.7  | 28      | 0.42 | 80  | Distribución sesgada; imputación requerida |
| Fare     | 32.20 | 14.45   | 0    | 512 | Cola derecha; aplicar log-transform          |
| Pclass   | –    | –      | 1    | 3   | Distribución inversa con supervivencia      |
| Sex      | –    | –      | –   | –  | Sexo femenino tiene alta supervivencia       |
| Embarked | –    | –      | –   | –  | La mayoría embarcó en S                    |

### 5.2 Valores faltantes

| Variable | % NA  | Estrategia                                  |
| -------- | ----- | ------------------------------------------- |
| Cabin    | 77%   | Crear indicador binario                     |
| Age      | 19.8% | Imputación supervisada (median / KNN / RF) |
| Embarked | 0.22% | Imputación por moda                        |

### 5.3 Outliers y normalidad

* **Fare:** outliers informativos → log-transform
* **Age:** distribución sesgada → imputación y escalado

#### Test de normalidad

| Variable | Shapiro-Wilk p | KS p   | Interpretación                                       |
| -------- | -------------- | ------ | ----------------------------------------------------- |
| Age      | <0.001         | 0.123  | No normal, efecto débil                              |
| Fare     | <0.001         | <0.001 | No normal, diferencias significativas según Survived |

### 5.4 Correlaciones

| Variable | Métrica    | Valor | Interpretación                |
| -------- | ----------- | ----- | ------------------------------ |
| Age      | Pearson r   | 0.06  | Débil correlación            |
| Fare     | Pearson r   | 0.26  | Correlación moderada          |
| Pclass   | Spearman ρ | -0.34 | Relación inversa con Survived |
| Sex      | Cramér V   | 0.54  | Asociación fuerte             |
| Embarked | Cramér V   | 0.10  | Asociación débil             |

---

## 6. Visualización avanzada

```python
# Distribución y supervivencia
sns.countplot(data=train, x="Sex", hue="Survived")
sns.countplot(data=train, x="Pclass", hue="Survived")
sns.kdeplot(data=train, x="Age", hue="Survived", fill=True)
sns.boxplot(data=train, x="Survived", y="Fare")
```

![Survival by Sex and Pclass](./images/survival_by_sex.png)

---

## 7. Conclusiones del EDA

* Sexo y clase son determinantes de supervivencia.
* `Fare` refleja capacidad económica; variable predictiva.
* Edad muestra efecto moderado; requiere imputación.
* `Cabin` ausente/presente es informativa.
* La estructura del dataset refleja inequidades socioeconómicas y patrones de supervivencia no lineales.

---

## 8. Próximos pasos técnicos

1. Ingeniería avanzada de variables derivadas y categóricas.
2. Construcción de pipelines reproducibles: imputación, escalado y codificación.
3. Entrenamiento y ajuste de modelos supervisados: Decision Tree, KNN, Random Forest.
4. Evaluación rigurosa mediante cross-validation y métricas robustas.
5. Predicciones finales listas para envío a Kaggle.

---

## 9. Referencias técnicas

* [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
* Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011
* Seaborn: [https://seaborn.pydata.org](https://seaborn.pydata.org)
* SciPy Stats: [https://docs.scipy.org](https://docs.scipy.org)
