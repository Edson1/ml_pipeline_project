#!/usr/bin/env python3
import aws_cdk as cdk
from infrastructure.ml_pipeline_stack import MLPipelineStack

app = cdk.App()
MLPipelineStack(app, "MLPipelineStack")
app.synth()
