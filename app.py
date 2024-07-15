import streamlit as st
import os
import numpy as np
from descriptor import glcm, bitdesc
from app_distance import calculate_similarity

# Function to load signatures based on selected descriptor
def load_signatures(desc_type):
    if desc_type == 'GLCM':
        return np.load('signatures_glcm.npy', allow_pickle=True)
    elif desc_type == 'BiT':
        return np.load('signatures_bit.npy', allow_pickle=True)
    else:
        return None

# Configure Streamlit theme and layout
st.markdown("""
    <style>
    body, .main, .sidebar, .stApp {
        background-color: #000000; /* Black background color */
        color: #ffd700; /* Gold text color */
        font-family: 'Arial', sans-serif; /* Font family */
    }
    .stButton>button {
        background-color: #ffd700;
        color: #000000;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease; /* Smooth transition for hover effect */
    }
    .stButton>button:hover {
        background-color: #e6c600;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #333333;
        color: #ffd700;
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 5px rgba(255, 215, 0, 0.6); /* Light gold shadow */
    }
    .stSelectbox select {
        background-color: #333333;
        color: #ffd700;
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 5px rgba(255, 215, 0, 0.6); /* Light gold shadow */
    }
    .stText {
        color: #ffd700;
        font-size: 18px;
        margin: 20px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8); /* Text shadow effect */
    }
    .stHeader {
        color: #ffd700;
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #666666;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8); /* Text shadow effect */
    }
    .gold-title {
        font-size: 32px;
        background: -webkit-linear-gradient(#ffd700, #e6c600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    .stImage {
        border: 2px solid #666666;
        border-radius: 10px;
        padding: 10px;
        background-color: #2e2e2e;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.8); /* Image shadow effect */
        transition: transform 0.3s ease; /* Smooth transition for hover effect */
    }
    .stImage:hover {
        transform: scale(1.05); /* Scale effect on hover */
    }
    .stSidebar {
        background-color: #000000;
        color: #ffd700;
    }
    .stSidebar > .stSubheader {
        color: #ffd700;
    }
    .stSidebar .stSelectbox select, .stSidebar .stTextInput input, .stSidebar .stTextArea textarea {
        background-color: #333333;
        color: #ffd700;
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 0 5px rgba(255, 215, 0, 0.6); /* Light gold shadow */
    }
    .stFooter {
        text-align: center;
        padding: 10px;
        background-color: #2e2e2e;
        color: #ffd700;
        border-top: 2px solid #666666;
    }
    </style>
    """, unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.markdown('<h1 class="gold-title">Content-Based Image Retrieval</h1>', unsafe_allow_html=True)
st.write('This App retrieves images based on their content using GLCM and BiT descriptors.')

# Sidebar parameters
st.sidebar.header('Parameters')
num_images = st.sidebar.number_input('Number of similar images to display', min_value=1, max_value=10, value=5)
distance_measure = st.sidebar.selectbox('Select distance measure', ['Euclidean', 'Manhattan', 'Chebyshev', 'Canberra'])
descriptor_selected = st.sidebar.selectbox('Select descriptor', ['GLCM', 'BiT'])

# Load signatures from the database
signatures = load_signatures(descriptor_selected)

# Image upload section
st.header('Upload an Image')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True, clamp=True)
    
    # Extract features from the uploaded image based on the selected descriptor
    if descriptor_selected == 'GLCM':
        features = glcm("temp_image.png")
    elif descriptor_selected == 'BiT':
        features = bitdesc("temp_image.png")
    else:
        features = None
    
    if features is None:
        st.error("Error extracting features. Please try uploading a different image.")
    elif len(signatures) == 0:
        st.error("No features found in the database. Please ensure the dataset has been processed.")
    else:
        # Calculate similarities and retrieve similar images
        similar_images = calculate_similarity(signatures, features, distance_measure, num_images)
        
        # Display similar images
        if similar_images:
            st.header('Similar Images')
            for img_path in similar_images:
                img_abs_path = os.path.abspath(os.path.join('./Projet1_Dataset/Projet1_Dataset', img_path))
                if os.path.isfile(img_abs_path):
                    st.image(img_abs_path, caption=os.path.basename(img_abs_path), use_column_width=True, clamp=True)
                else:
                    st.warning(f"Cannot open image {img_path}. File not found.")
        else:
            st.info("No similar images found.")

# Footer or additional information section
st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️ by [Youssef jaber]")

# Footer section
st.markdown("""
    <div class="stFooter">
        <p>© Teccart 2024 Powered by Streamlit and Python</p>
    </div>
    """, unsafe_allow_html=True)
