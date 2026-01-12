from uav.setup.download_data import download_and_extract
from uav.setup.process_data import process_dataset

def automated_setup(verbose: bool) -> None:
    """Automatically downloads and processes the Anti-UAV300 dataset."""
    print("Running automated setup.")

    print("Downloading...")    
    success = download_and_extract(
        url="https://drive.google.com/uc?id=1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr", # Dataset provided via GitHub: https://github.com/ucas-vg/Anti-UAV
        zip_path="datasets/anti-uav300-raw.zip",
        extract_dir="datasets/anti-uav300-raw",
        verbose=verbose,
        remove_zip = True
    )
    
    if not success:
        print("Download failed.")

    print("Formatting...")
    success = process_dataset(
        source_dir = "datasets/anti-uav300-raw",
        target_dir = "datasets/anti-uav300",
        verbose=verbose,
        remove_source = True,
    )

    if not success:
        print("Dataset formatting failed.")

    print("Done!")
