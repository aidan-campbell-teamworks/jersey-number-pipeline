import argparse
import cv2
import os
import random

def visualize_legibility(frame):
    print("\n=================================================")
    print("Legibility Approval/Rejection Instructions:")
    print("y: legible")
    print("n: illegible")
    print("q: quit")
    print("=================================================\n")

    key = ""
    while key.lower() not in ["y", "n"]:
        cv2.imshow("Legibility Label", frame)
        key = chr(cv2.waitKey(0))
    cv2.destroyAllWindows()

    if key == "y":
        return True
    elif key == "n":
        return False
    elif key == "q":
        raise KeyboardInterrupt

def main(num_images=None):
    data_dir = "../data/Football/jnp/train"
    img_dir = os.path.join(data_dir, "images")
    out_dir = os.path.join(data_dir, "legibility_data/images")
    os.makedirs(out_dir, exist_ok=True)
    outfile = open(os.path.join(data_dir, "legibility_data/legibility_football_gt.txt"), "w")
    files = os.listdir(img_dir)
    random.shuffle(files)
    if num_images is None:
        files = files[:num_images]

    for i, img_file in enumerate(files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        is_legible = visualize_legibility(img)

        cv2.imwrite(os.path.join(out_dir, img_file), img)
        outfile.write(f"{img_file},{int(is_legible)}\n")
        print(f"Processed {i+1} images")
    outfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, required=False)
    args = parser.parse_args()

    main(args.num_images)