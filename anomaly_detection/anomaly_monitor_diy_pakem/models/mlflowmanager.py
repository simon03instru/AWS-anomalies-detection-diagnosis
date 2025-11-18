import mlflow
import os

class MLflowManager:
    def __init__(self):
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.job_id = os.getenv("MLFLOW_JOB_ID", "default_job_id")

        # Set MLflow artifact upload/download timeout (in seconds)
        # Default to 1 hour for large GGUF files (typically 2-20GB)
        timeout = os.getenv("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "3600")
        os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = timeout
        print(f"MLflow artifact timeout set to {timeout} seconds ({int(timeout)//60} minutes)")

    def start_experiment(self):
        """Start MLflow experiment."""
        # Create MLflow directory if it doesn't exist
        if self.tracking_uri == "./mlruns":
            if not os.path.exists(self.tracking_uri):
                os.makedirs(self.tracking_uri, exist_ok=True)

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow_job_id = self.job_id
        mlflow.enable_system_metrics_logging()

        experiment_name = f"experiment_agent_{mlflow_job_id}"

        # Check if the experiment already exists
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment:
            self.experiment_id = experiment.experiment_id  # Use existing experiment
        else:
            self.experiment_id = mlflow.create_experiment(name=experiment_name)  # Create new one

        mlflow.start_run(run_name=f"run_{mlflow_job_id}", experiment_id=self.experiment_id)

        # Autolog
        mlflow.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
        )

    def log_metric(self, metric_name, metric_value, step=0):
        try:
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value, step)
        except Exception as e:
            print(f"Warning: Failed to log metric {metric_name}: {e}")

    def log_param(self, param_name, param_value):
        try:
            mlflow.log_param(param_name, param_value)
        except Exception as e:
            print(f"Warning: Failed to log parameter {param_name}: {e}")

    def end_experiment(self):
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Warning: Failed to end MLflow run: {e}")

    # Log the model to MLflow: based on pytorch, 
    # if you want to log the model based on other framework, you need to implement a new function
    def log_model_pytorch(self, model, artifact_path, registered_model_name=None):
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            registered_model_name=registered_model_name
        )

    def log_artifact(self, local_path, artifact_path=None):
        try:
            # Get file size for logging
            if os.path.isfile(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"📤 Uploading {os.path.basename(local_path)} ({file_size_mb:.1f} MB) to MLflow...")

            mlflow.log_artifact(local_path, artifact_path)
            print(f"✅ Successfully logged artifact: {artifact_path or os.path.basename(local_path)}")
        except Exception as e:
            print(f"❌ Failed to log artifact {local_path}: {e}")
            # For large files, suggest timeout increase
            if os.path.isfile(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                file_size_gb = file_size_mb / 1024

                # GGUF files are typically 2-20GB+, provide specific guidance
                if file_size_mb > 500:  # Files larger than 500MB
                    if ".gguf" in local_path.lower():
                        # GGUF-specific timeout recommendations
                        if file_size_gb < 2:
                            recommended_timeout = 1800  # 30 minutes
                        elif file_size_gb < 5:
                            recommended_timeout = 3600  # 1 hour
                        elif file_size_gb < 10:
                            recommended_timeout = 7200  # 2 hours
                        else:
                            recommended_timeout = 10800  # 3 hours

                        print(f"💡 Large GGUF file ({file_size_gb:.1f} GB) detected.")
                        print(f"   Recommended timeout: {recommended_timeout} seconds ({recommended_timeout//60} minutes)")
                        print(f"   Set with: export MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT={recommended_timeout}")
                    else:
                        # General large file guidance
                        recommended_timeout = max(1800, int(file_size_mb * 2))  # 2 seconds per MB, minimum 30 min
                        print(f"💡 Large file ({file_size_mb:.1f} MB) detected.")
                        print(f"   Consider increasing timeout: export MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT={recommended_timeout}")

    def get_artifact_download_info(self):
        """Get information about how to download artifacts"""
        try:
            run_id = mlflow.active_run().info.run_id
            tracking_uri = mlflow.get_tracking_uri()
            return {
                "run_id": run_id,
                "tracking_uri": tracking_uri,
                "download_command": f"mlflow artifacts download --run-id {run_id}"
            }
        except Exception as e:
            print(f"Warning: Could not get artifact download info: {e}")
            return None