import cv2
import os

def visualize_legibility(frame):
    print("\n=================================================")
    print("Legibility Approval/Rejection Instructions:")
    print("y: legible")
    print("n: illegible")
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

def main():
    data_dir = "../data/Football/jnp/train"
    img_dir = os.path.join(data_dir, "images")
    out_dir = os.path.join(data_dir, "legibility_data/images")
    os.makedirs(out_dir, exist_ok=True)
    outfile = open(os.path.join(data_dir, "legibility_data/legibility_football_gt.txt"), "w")

    for i, img_file in enumerate(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, img_file))
        is_legible = visualize_legibility(img)

        cv2.imwrite(os.path.join(out_dir, img_file), img)
        outfile.write(f"{img_file},{int(is_legible)}\n")
        print(f"Processed {i+1} images")
        # if i == 10:
        #     break
    outfile.close()

if __name__ == "__main__":
    main()