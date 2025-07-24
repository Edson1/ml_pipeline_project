from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    RemovalPolicy,
    CfnOutput
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

        # Rol de ejecución para SageMaker
        role = iam.Role(self, "SageMakerExecutionRole",
                assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
                managed_policies=[
                    iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
                ]
        )

        # Política para permitir acceso a el bucket creado: GetObject, PutObject, ListBucket
        bucket.grant_read_write(role) 

        CfnOutput(self, "S3BucketName", value=bucket.bucket_name)
        CfnOutput(self, "ExecutionRoleArn", value=role.role_arn)
