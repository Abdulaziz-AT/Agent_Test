import os
import comet_ml

PROJECT_NAME = "Chatbot 2 Proj"
EXPERIMENT_NAME = "A_2.015_meta-llama/llama-4-maverick_gemini/gemini-2.5-pro_20250720_1512"

def download_experiment_assets(project_name, experiment_name):
    """
    Connects to Comet/Opik using local credentials and downloads assets from an experiment.

    Args:
        project_name (str): The name of the project containing the experiment.
        experiment_name (str): The name of the experiment to download.
    """
    print("Attempting to initialize Comet API using local configuration...")

    try:
        api = comet_ml.API()
    except Exception as e:
        print("\nFATAL ERROR: Could not initialize the Comet API.")
        print("Please ensure you have run 'comet init' in your terminal to create a .comet.config file.")
        print(f"Details: {e}")
        return

    print("API initialized successfully.")

    experiment_path = f"{project_name}/{experiment_name}"
    print(f"Attempting to fetch experiment: {experiment_path}")

    try:
        experiment = api.get(experiment_path)
    except Exception as e:
        print(f"Error: Could not find the experiment. Please verify the project and experiment names.")
        print(f"Details: {e}")
        return

    if not experiment:
        print("Error: Experiment not found. Please check your Project Name and Experiment Name.")
        return

    print(f"Successfully found experiment '{experiment.name}'.")

    # Create a safe directory name by replacing '/' with '_'
    safe_dir_name = experiment_name.replace("/", "_")
    output_dir = f"./{safe_dir_name}_assets"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Assets will be saved to: {output_dir}")

    asset_list = experiment.get_asset_list()
    if not asset_list:
        print("No assets found for this experiment.")
        return

    print(f"Found {len(asset_list)} assets. Starting download...")

    for asset in asset_list:
        asset_id = asset['assetId']
        file_name = asset['fileName']
        file_path = os.path.join(output_dir, file_name)

        print(f"Downloading: {file_name}...")
        try:
            # Check if the asset exists before downloading
            if experiment.get_asset(asset_id):
                asset_data = experiment.get_asset(asset_id, return_type="binary")
                with open(file_path, "wb") as f:
                    f.write(asset_data)
                print(f" -> Successfully saved to {file_path}")
            else:
                print(f" -> Asset {file_name} not found.")
        except Exception as e:
            print(f" -> Failed to download {file_name}. Error: {e}")

    print("\nDownload process finished.")

if __name__ == "__main__":
    download_experiment_assets(
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME
    )