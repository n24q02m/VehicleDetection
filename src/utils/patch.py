import os
import shutil
from pathlib import Path


def patch_ultralytics():
    """
    Patch Ultralytics package with custom modifications from patches directory.
    Works both locally and on Kaggle.
    """
    try:
        # Detect if running on Kaggle
        on_kaggle = "KAGGLE_URL_BASE" in os.environ

        if on_kaggle:
            print("Running on Kaggle - Applying patches.")
        else:
            print("Running locally - Applying patches.")

        # Find ultralytics package location
        import ultralytics

        ultralytics_path = Path(ultralytics.__file__).parent

        # Define patch mappings
        patches = {
            "task.py": ultralytics_path / "nn" / "tasks.py",
            "modules_init.py": ultralytics_path / "nn" / "modules" / "__init__.py",
            "conv.py": ultralytics_path / "nn" / "modules" / "conv.py",
        }

        # Determine which check file to use based on the script name
        script_name = Path(__file__).stem
        if script_name == "finetune":
            patches["finetune_check.py"] = ultralytics_path / "utils" / "checks.py"
        elif script_name == "train":
            patches["train_check.py"] = ultralytics_path / "utils" / "checks.py"

        # Get patches directory path relative to this script
        patches_dir = Path(__file__).parent.parent.parent / "patches"

        # Apply patches
        print("Applying patches to Ultralytics package...")
        for patch_file, target_path in patches.items():
            patch_path = patches_dir / patch_file
            if patch_path.exists():
                try:
                    # Rename patch file to match the target file name
                    temp_patch_path = patch_path.with_name(target_path.name)
                    shutil.copy2(patch_path, temp_patch_path)
                    shutil.copy2(temp_patch_path, target_path)
                    temp_patch_path.unlink()  # Remove the temporary renamed file
                    print(f"Patched {target_path}")
                except PermissionError:
                    if on_kaggle:
                        print(
                            f"Permission denied: Cannot patch {target_path} on Kaggle. Skipping this patch."
                        )
                    else:
                        raise
            else:
                print(f"Warning: Patch file {patch_path} not found")

        print("Patches applied successfully.")

    except ImportError:
        print("Warning: Ultralytics package not found")
    except Exception as e:
        print(f"Error applying patches: {e}")
