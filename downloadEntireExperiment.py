import os
import comet_ml
import logging

# Configure logging to provide more detailed error messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        # Initialize the Comet API with a retry mechanism to handle temporary connection issues
        api = comet_ml.API()
        api.max_retries = 3
    except Exception as e:
        logging.error("FATAL ERROR: Could not initialize the Comet API.")
        logging.error("Please ensure you have run 'comet init' in your terminal to create a .comet.config file.")
        logging.error(f"Details: {e}")
        return

    print("API initialized successfully.")

    # Construct the experiment path and fetch the experiment with error handling
    experiment_path = f"{project_name}/{experiment_name}"
    print(f"Attempting to fetch experiment: {experiment_path}")

    try:
        experiment = api.get(experiment_path)
    except Exception as e:
        logging.error(f"Error: Could not find the experiment. Please verify the project and experiment names.")
        logging.error(f"Details: {e}")
        return

    if not experiment:
        logging.error("Error: Experiment not found. Please check your Project Name and Experiment Name.")
        return

    print(f"Successfully found experiment '{experiment.name}'.")

    # Create a safe directory name and output directory for the assets
    safe_dir_name = experiment_name.replace("/", "_")
    output_dir = f"./{safe_dir_name}_assets"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Assets will be saved to: {output_dir}")

    # Get the asset list and handle the case where no assets are found
    asset_list = experiment.get_asset_list()
    if not asset_list:
        logging.info("No assets found for this experiment.")
        return

    print(f"Found {len(asset_list)} assets. Starting download...")

    # Download each asset with a retry mechanism and error handling
    for asset in asset_list:
        asset_id = asset['assetId']
        file_name = asset['fileName']
        file_path = os.path.join(output_dir, file_name)

        print(f"Downloading: {file_name}...")
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                asset_data = experiment.get_asset(asset_id, return_type="binary")
                with open(file_path, "wb") as f:
                    f.write(asset_data)
                print(f" -> Successfully saved to {file_path}")
                break
            except Exception as e:
                logging.error(f" -> Failed to download {file_name}. Error: {e}")
                retries += 1
                if retries < max_retries:
                    print(f" -> Retrying download of {file_name}...")
                else:
                    logging.error(f" -> Failed to download {file_name} after {max_retries} retries.")
                    break

    print("\nDownload process finished.")


if __name__ == "__main__":
    download_experiment_assets(
        project_name=PROJECT_NAME,
        experiment_name=EXPERIMENT_NAME
    )