from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.model import SKLearnModel

def register_model(model_path, role, model_package_group_name):
    sagemaker_session = PipelineSession()

    model = SKLearnModel(
        model_data=model_path,
        role=role,
        entry_point="inference.py",
        framework_version="0.23-1",
        sagemaker_session=sagemaker_session
    )

    # Register the model in the SageMaker Model Registry with inference instance and transform job: ml.t2.medium ml.m5.large
    model.register(content_types=["text/csv"], response_types=["text/csv"],
                   inference_instances=["ml.m4.xlarge"],
                   transform_instances=["ml.m4.xlarge"],
                   model_package_group_name=model_package_group_name)
