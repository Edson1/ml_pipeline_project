# ML Pipeline with SageMaker and AWS CDK

This project uses SageMaker Pipelines, AWS CDK, and Python to orchestrate the ML lifecycle for the Titanic dataset.

# 1. Install dependencies 
pip install -r requirements.txt

# 2. Inicializa el entorno de CDK (solo la primera vez):
cdk bootstrap

# 3. Despliega la infraestructura:
cdk deploy

# Verifica el estado del pipeline desde la consola de SageMaker.
