import cv2
import os
import numpy as np
from descriptor import glcm, bitdesc

def extract_features(image_path, descriptor):
    img = cv2.imread(image_path, 0)
    if img is not None:
        try:
            features = descriptor(img)
            return features
        except Exception as e:
            print(f"Error extracting features from {image_path} with {descriptor.__name__}: {e}")
            return None
    else:
        print(f"Failed to read image {image_path}")
        return None

descriptors = [glcm, bitdesc]

def process_datasets(root_folder):
    all_features_glcm = []
    all_features_bit = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                
                # Features for GLCM
                features_glcm = extract_features(image_rel_path, glcm)
                if features_glcm is not None:
                    features_glcm.extend([folder_name, relative_path])
                    all_features_glcm.append(features_glcm)
                    print(f"Processed {image_rel_path} with GLCM")
                
                # Features for BiT
                features_bit = extract_features(image_rel_path, bitdesc)
                if features_bit is not None:
                    features_bit.extend([folder_name, relative_path])
                    all_features_bit.append(features_bit)
                    print(f"Processed {image_rel_path} with BiT")
    
    if all_features_glcm and all_features_bit:
        signatures_glcm = np.array(all_features_glcm)
        np.save('signatures_glcm.npy', signatures_glcm)
        print('Successfully stored GLCM features in signatures_glcm.npy!')
        
        signatures_bit = np.array(all_features_bit)
        np.save('signatures_bit.npy', signatures_bit)
        print('Successfully stored BiT features in signatures_bit.npy!')
    else:
        print('No features were extracted.')

# Utilisation
process_datasets('./Projet1_Dataset/Projet1_Dataset')