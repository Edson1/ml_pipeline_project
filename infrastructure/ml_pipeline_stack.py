from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_sagemaker_alpha as sm_alpha,
    RemovalPolicy,
    Duration
)

from constructs import Construct

class MLPipelineStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        # S3 Bucket for input (upload to raw/train.csv) and output data
        bucket = s3.Bucket(self, "TitanicDataBucket",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # Role for SageMaker to access S3
        role = iam.Role(self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
            ]
        )

        # SageMaker Pipeline step1 with Preprocessing instance. ml.m5.xlarge
        preprocessing_step = sm_alpha.ProcessingStep(self, "PreprocessingStep",
            step_name="TitanicPreprocessing",
            processor=sm_alpha.SKLearnProcessor(
                framework_version="0.23-1",
                instance_type="ml.t3.medium",
                instance_count=1,
                role=role
            ),
            input_data={"train": sm_alpha.ProcessingInput.from_s3(bucket.s3_url_for_object("raw/train.csv"))},
            output_data={"output": sm_alpha.ProcessingOutput(output_name="processed", source="/opt/ml/processing/output")},
            code="pipeline/processing/preprocessing.py"
        )

        # SageMaker Pipeline step2 with Training instance
        # This step will use the output from the preprocessing step
        training_step = sm_alpha.TrainingStep(self, "TrainingStep",
            step_name="TitanicTraining",
            estimator=sm_alpha.SKLearnEstimator(
                entry_point="pipeline/training/train.py",
                instance_type="ml.m4.xlarge",
                instance_count=1,
                framework_version="0.23-1",
                role=role
            ),
            input_data={"train": preprocessing_step.outputs["output"]}
        )

        # SageMaker Pipeline step3: Register the trained model in SageMaker Model Registry
        model_step = sm_alpha.ModelStep(self, "RegisterModelStep",
            step_name="RegisterTitanicModel",
            model=sm_alpha.Model(
                role=role,
                image_uri=sm_alpha.SKLearnProcessor.get_image_uri("us-east-1", "0.23-1"),
                model_data=training_step.output_model_artifact
            ),
            model_package_group_name="TitanicModelGroup"
        )

        # Define the SageMaker Pipeline with all steps
        sm_alpha.Pipeline(self, "TitanicPipeline",
            pipeline_name="TitanicPipeline",
            steps=[preprocessing_step, training_step, model_step]
        )
