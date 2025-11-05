import os
from dotenv import load_dotenv
import comet_ml

# Load environment variables
load_dotenv()

# Initialize Comet experiment
experiment = comet_ml.Experiment(
    api_key=os.getenv("COMET_API_KEY"),
    project_name=os.getenv("COMET_PROJECT_NAME"),
    workspace=os.getenv("COMET_WORKSPACE")
)

# Log a test metric
experiment.log_parameter("test_param", "hello_comet")
experiment.log_metric("test_metric", 0.95)

print("✅ Comet ML connection successful!")
print(f"📊 View experiment at: {experiment.url}")

experiment.end()
