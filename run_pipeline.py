import boto3
from botocore.exceptions import ClientError
import logging, time
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep #, RegisterModel, ModelStep, EndpointConfigStep, EndpointStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
#from sagemaker.pipeline import PipelineModel
from sagemaker.inputs import TrainingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.parameters import ParameterString
from sagemaker.model import Model
#from sagemaker.predictor import Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get CDK outputs from the CloudFormation stack
def get_cdk_outputs(stack_name: str):
    cf = boto3.client("cloudformation")
    response = cf.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0].get("Outputs", [])
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}

cdk_outputs = get_cdk_outputs("MLPipelineStack")

bucket = cdk_outputs.get("S3BucketName")
role = cdk_outputs.get("ExecutionRoleArn")

prefix = "pipeline" #s3 folder for pipeline outputs

print(f"S3 Bucket Name: {bucket}")
print(f"SageMaker Execution Role ARN: {role}")

# Crear un cliente de SageMaker
sagemaker_client = boto3.client("sagemaker")

# Crear una sesión de SageMaker Pipeline
pipeline_session = PipelineSession()

# Parámetros del pipeline
input_data = ParameterString(name="InputDataUrl", default_value=f"s3://{bucket}/raw/train.csv")
model_output = ParameterString(name="ModelOutput", default_value=f"s3://{bucket}/ml/model")

# SageMaker Pipeline step1 with Preprocessing instance. ml.m5.xlarge not valid for precessing job
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="titanicpreprocessing",
    sagemaker_session=pipeline_session
)

# The preprocessing step will output a processed CSV file to the specified S3 bucket
# The preprocessing.py "output" (processed.csv) will be used as input for the training step
step_process = ProcessingStep(
    name="TitanicPreprocessing",
    processor=sklearn_processor,
    inputs=[ProcessingInput(input_name="input", source=input_data, destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(output_name="output",  source="/opt/ml/processing/output", destination=f"s3://{bucket}/processed.csv")],
    code="pipeline/processing/preprocessing.py"
)

# SageMaker Pipeline step2 with Training instance and train.py. This step will use the output (processed.csv) from the preprocessing step
sklearn_estimator = SKLearn(
    entry_point="pipeline/training/train.py",
    role=role,
    instance_type="ml.m4.xlarge",
    framework_version="0.23-1",
    base_job_name="titanictraining",
    sagemaker_session=pipeline_session
)

#trained model is saved in S3://sagemaker-..../pipelines-.....-TitanicTraining-..../output/model.tar.gz 
step_train = TrainingStep(
    name="TitanicTraining",
    estimator=sklearn_estimator,
    inputs={"train": TrainingInput(
        s3_data=step_process.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri, 
        content_type="text/csv"
    )}
)

# SageMaker Pipeline step3: Register the trained model artifact in SageMaker Model Registry 
model = Model(
    image_uri=sklearn_estimator.training_image_uri(),
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    sagemaker_session=pipeline_session,
    role=role,
    entry_point="pipeline/inference/inference.py"
)

#Create Model Registry with inference function from inference.py 
#Inference Model registered in Container1 arn:aws:sagemaker:...:model/pipelines-...-TitanicModelRegistra-...
step_register = CreateModelStep(
    name="TitanicModelRegistration",
    model=model
)

# Define the SageMaker Pipeline with all steps
pipeline = Pipeline(
    name="TitanicPipeline",
    parameters=[input_data, model_output],
    steps=[step_process, step_train, step_register],
    sagemaker_session=pipeline_session
)

# Function to upload raw data to S3 bucket
def upload_raw_data(bucket_name, local_path, s3_prefix="raw/train.csv"):
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket_name, s3_prefix)
    print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_prefix}")

if __name__ == "__main__":
    # Upload the raw data to S3 bucket
    upload_raw_data(bucket, "train.csv")
    time.sleep(30)

    # Create or update the pipeline
    pipeline.upsert(role_arn=role)

    #print(pipeline.definition())
    #print("Pipeline execution summary of previous executions:")
    #print(pipeline.list_executions())

    print("------Describe pipeline------")
    print(pipeline.describe())
    
    # Start the pipeline execution
    execution = pipeline.start()

    execution_arn = execution.arn
    logger.info(f"Pipeline execution (ARN): {execution_arn}")

    # execution.wait() no muestra errores especificos cuando falla algun step
    #execution.wait(delay=30, max_attempts=60)

    # Validate the execution progress with own waiting loop
    description = execution.describe()
    status = description["PipelineExecutionStatus"]

    while status == "Executing":
        logger.info("Waiting: Esperando a que la ejecución del pipeline finalice...")
        time.sleep(30)

        description = execution.describe()
        status = description["PipelineExecutionStatus"]

    logger.info(f"Estado final: {status}")

    if status != "Succeeded":
        logger.error(f"Ejecución fallida: {description.get('FailureReason')}")

    logger.info("Pasos ejecutados del pipeline:")
    steps = list(execution.list_steps())
    if not steps:
        logger.warning("No se ejecutó ningún paso. Verifica la definición del pipeline o errores previos.")
    else:
        for step in steps:
            logger.info(f" - {step['StepName']} => {step['StepStatus']}")


    print("-----List all steps executed in the pipeline, EMPTY list if nothing executed------")
    print(execution.list_steps())

    print("----------------PipelineParameters list------")
    print(execution.list_parameters())

    print("----------------Describe execution------")
    print(execution.describe())

    # Get the model name from the step_register to create endpoint
    model_name = None

    # If the model was not registered, use the last registered model in the Model Registry
    try:
        model_name = str(step_register.properties.ModelName) #.to_string()
        logger.info(f"Model registered: {model_name}")
    except Exception as e:
        logger.info(f"Using the last registered model in the Model Registry")
        model_name = "pipelines-5d7th42m1xus-TitanicModelRegistra-GJ7rBsDBhz"    

    endpoint_config_name = "TitanicEndpointConfig"
    endpoint_name = "TitanicInferenceEndpoint"

    # Crear configuración del endpoint
    try:
        print(f"Creating endpoint config: {endpoint_config_name}")

        sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                    "InitialVariantWeight": 1
                }
            ]
        )
    except:
        print(f"Using Endpoint config '{endpoint_config_name}' that already exists.")


    print(sagemaker_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name))

    # Crear o actualizar el endpoint
    existing_endpoints = sagemaker_client.list_endpoints(NameContains=endpoint_name)["Endpoints"]

    if any(ep["EndpointName"] == endpoint_name for ep in existing_endpoints):
        print(f"Updating endpoint: {endpoint_name}")
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        print(f"Creating endpoint: {endpoint_name}")

        try:
            response = sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print("Endpoint creation started:", response)
            #fails if instance type (like ml.m4.xlarge) is not available or you’ve hit a service quota
        except ClientError as e:
            logging.error(e.response["Error"]["Message"])




