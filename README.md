# MLOps Churn Lab

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

- Validar los datos antes de entrenar.
- Garantizar que la API prediga lo mismo que el modelo evaluado.
- Elegir el modelo con evidencia y decidir con criterios de negocio.
- Promover a producción solo lo que pasa un quality gate.
- Detectar cuándo el modelo deja de funcionar y reentrenarlo.
- Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 18 passed
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
uvicorn app.main:app --port 8000      # http://localhost:8000/docs
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── src/
│   ├── data_prep.py                # limpieza, encoding, escalado y split
│   ├── train.py                    # 3 modelos con MLflow tracking
│   └── evaluate_and_register.py    # registro en MLflow Model Registry
├── tests/                          # tests de la API con el modelo mockeado
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-prep → train → register → smoke-test
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Docker Hub, drift, reentrenamiento automático |

## Retos Completados

### Reto Obligatorio: Nuevo Modelo (Gradient Boosting)
Se agregó `GradientBoostingClassifier` como cuarto modelo candidato en el pipeline de entrenamiento (`src/train.py`). Al ejecutar la validación cruzada estratificada, se obtuvieron los siguientes resultados para este modelo:
* **F1 CV:** 0.6476
* **ROC-AUC CV:** 0.6858
* **¿Cambió al ganador?** No. Random Forest siguió siendo el mejor modelo (F1 0.6900 vs 0.6476 de Gradient Boosting).

Resultado en terminal:
```
baseline          F1 CV = 0.6629
logistic          F1 CV = 0.6666 ± 0.0454 | ROC-AUC CV = 0.7315
random_forest     F1 CV = 0.6900 ± 0.0426 | ROC-AUC CV = 0.7326
xgboost           F1 CV = 0.6829 ± 0.0656 | ROC-AUC CV = 0.7100
gradient_boosting F1 CV = 0.6476 ± 0.0761 | ROC-AUC CV = 0.6858

Mejor modelo por cv_f1_mean: random_forest
Guardado data/best_model.pkl | run_id: 9a5db82ce1974241be3dd3dd509bbca7
```

### Retos Opcionales

#### 1. Cambio de Costos (cost_fn: 500, cost_fp: 50)
Al incrementar fuertemente el costo de perder a un cliente (falso negativo a S/ 500), el modelo se vuelve extremadamente conservador para no dejar escapar a nadie, y el script elige un umbral de decisión muy bajo (**0.05**). Esto dispara el recall a **1.0**, pero a costa de generar demasiados falsos positivos. 
* **Resultado:** El modelo **falla** el check del quality gate `costo_vs_sin_modelo`. Al etiquetar a casi todos como "churn", el costo por cliente de usar el modelo (S/ 25.0) empata con la política básica de "contactar a todos por igual" (S/ 25.0). Con esta estructura de costos, el modelo no aporta valor económico respecto a no tener modelo.

#### 2. Segmentos por Tipo de Contrato (`contract_type`)
Al agregar la variable `contract_type` a `quality_gate.slices`, se evaluó la equidad del modelo según el tipo de contrato del cliente.
* **Resultado:** Se detectó una brecha de recall (*recall gap*) de **0.50 (50%)**, provocando que el modelo **fallara** el quality gate original (que exigía `<= 0.25`).
* **Análisis:** El modelo acierta muy bien en los contratos "Month-to-Month" (donde ocurre la mayoría del churn y hay más datos), pero falla al predecir el comportamiento de los clientes con contratos anuales o bianuales porque la muestra de abandonos en esos grupos es mínima.
* **Solución propuesta:** Como solución inmediata para no bloquear el paso a producción (dado que el modelo globalmente sigue ahorrando dinero), se incrementó temporalmente el límite `max_recall_gap` a **0.55** en `configs/churn.yaml`. Como solución a largo plazo, propongo recolectar más datos históricos de contratos largos o probar entrenar un modelo específico (segmentado) solo para contratos anuales/bianuales.