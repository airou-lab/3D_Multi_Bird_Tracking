import os
import numpy
import argparse
import FeatureMatching as fm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Matching')
    parser.add_argument('img1_name', help='Name of the first image located in the "images" folder')
    parser.add_argument('img2_name', help='Name of the second image located in the "images" folder')
    parser.add_argument('bbox_name_1', help='Name of the bounded box file of the first image located in the "bounded_boxes" folder')
    parser.add_argument('bbox_name_2', help='Name of the bounded box file of the second image located in the "bounded_boxes" folder')
    args = parser.parse_args()

    result, kp1, kp2, matches = fm.match_features(args.img1_name, args.img2_name, args.bbox_name_1, args.bbox_name_2, show=True)
