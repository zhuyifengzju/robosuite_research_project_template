import h5py
import cv2
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

with h5py.File(args.dataset) as f:
    images = f["data/demo_0/obs/agentview_rgb"][()]
    depth_images = f["data/demo_0/obs/agentview_depth"][()]
for image in images:
    cv2.imshow("", image[..., ::-1])
    cv2.waitKey(10)
for image in depth_images:
    print(image.max(), image.min())
    cv2.imshow("", image)
    cv2.waitKey(10)    