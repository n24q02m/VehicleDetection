import os


def create_public_test_labels():
    # Create the output directory
    output_dir = "./data/public_test_labels"
    os.makedirs(output_dir, exist_ok=True)

    # Read the predict.txt file
    predict_file = "./runs/better-train-yolov8m-ghost-p2/test_predict.txt"
    with open(predict_file, "r") as f:
        lines = f.readlines()

    # Process each line
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            filename = parts[0]
            cls = parts[1]
            x_center = parts[2]
            y_center = parts[3]
            width = parts[4]
            height = parts[5]

            # Remove file extension and replace with .txt
            base_name = os.path.splitext(filename)[0]
            txt_filename = base_name + ".txt"
            txt_filepath = os.path.join(output_dir, txt_filename)

            # Write to the txt file
            with open(txt_filepath, "a") as txt_file:
                txt_file.write(f"{cls} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":
    create_public_test_labels()
