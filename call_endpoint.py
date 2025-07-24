import boto3
import json

runtime = boto3.client("sagemaker-runtime")

#Payload for a single passenger
payload = {
    "passenger": [33, 22.0, 1, 0, 7.25, 0, 1, 0, 0, 0, 0, 1] 
}
##get label 1 with:    [1,23.0,1,0,21228,82.2667,0,0,1,1,0,0]

# Invoke the endpoint with the JSON payload
response = runtime.invoke_endpoint(
    EndpointName="TitanicInferenceEndpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

print(response)

# Print the response from the endpoint
print("Prediction: Passenger: 1 = Survived, 0 = Not Survived >")
print(response["Body"].read().decode("utf-8"))
