# ML project on AWS SageMaker (CDK and Pipeline)
This project uses SageMaker Pipelines, AWS CDK, and Python to orchestrate the ML lifecycle for the Titanic dataset from Kaggle: 
https://www.kaggle.com/datasets/brendan45774/test-file/code

It preprocesses the data and trains a supervised classification model (SKLearn Estimator) on the Titanic dataset, registers the model, and deploys it as a real-time inference endpoint.

The project also has csv files in each pipeline folder for local testing of pipeline steps.

## Prerequisites
- Python 3.7+
- AWS CLI and AWS credentials configured
- Node.js & npm (for CDK)
- CDK installed: `npm install -g aws-cdk`
- Ensure your AWS account has SageMaker instance quotas (e.g., ml.m4.xlarge).
- SageMaker endpoint may incur cost when deployed, and also the SageMaker pipeline running steps.

## 1. Install dependencies 
pip install -r requirements.txt

## 2. Deploys the CDK toolkit stack into the AWS environment (just one time):
cdk bootstrap

## 3. Deploys the MLPipelineStack stack (ml_pipeline_stack.py) into your AWS account with CDK (validate it in AWS CloudFormation stacks) :
cdk deploy

---
It creates or updates the MLPipelineStack stack, which has output with S3 bucket name mlpipelinestack-titanicdatabucket... and the IAM role ARN (of MLPipelineStack-SageMakerExecutionRole...).

## 4. Run pipeline and deploy the model in SageMaker inference endpoint
python run_pipeline.py

---
It upload the Titanic dataset (train.csv file) to your S3 bucket in a folder called "raw". 
- {mlpipelinestack-titanicdatabucket...}/raw/train.csv

Verify the executed steps and pipeline resources from AWS SageMaker AI website:
- Processing / Processing Jobs
- Training / Training Jobs
- Inference / Models (registered)
- Inference / Endpoint configurations
- Inference / Endpoints

---
Logs are available in AWS CloudWatch Logs 

## 7. Test the AWS SageMaker inference endpoint (internally) using the sagemaker-runtime client:
python call_endpoint.py

---
Model requires 12 features for a single prediction. Edit the Payload in call_endpoint.py for other tests:
- [Sex,Age,SibSp,Parch,Ticket,Fare,Embarked_C,Embarked_Q,Embarked_S,Pclass_1,Pclass_2,Pclass_3]
- Request Payload example: [33, 22.0, 1, 0, 7.25, 0, 1, 0, 0, 0, 0, 1] 
---
Valid feature types:
{
    Sex: int,
    Age: float,
    SibSp: int,
    Parch: int,
    Ticket: int,
    Fare: float,
    Embarked_C: int,
    Embarked_Q: int,
    Embarked_S: int,
    Pclass_1: int,
    Pclass_2: int,
    Pclass_3: int
}

---
HTTP body response will return 1 or 0 for the request data:
Prediction: Passenger: 1 = Survived, 0 = Not Survived.

## external POST request to the AWS SageMaker endpoint will need a AWS token for API authorization:
https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/TitanicInferenceEndpoint/invocations

---
Useful commands: 
- cdk list --verbose  
- cdk destroy MLPipelineStack 
