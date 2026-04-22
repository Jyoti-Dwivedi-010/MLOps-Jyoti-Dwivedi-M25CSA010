import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import json
import os
from dataloader import get_transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(layout="wide", page_title="CityScape Segmentation")

# Color palette for segmentation classes (Cityscapes)
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [70, 70, 70],      # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170, 30],    # traffic light
    [220, 220, 0],     # traffic sign
    [107, 142, 35],    # vegetation
    [152, 251, 152],   # terrain
    [70, 130, 180],    # sky
    [220, 20, 60],     # person
    [255, 0, 0],       # rider
    [0, 0, 142],       # car
    [0, 0, 70],        # truck
    [0, 60, 100],      # bus
    [0, 80, 100],      # train
    [0, 0, 230],       # motorcycle
    [119, 11, 32],     # bicycle
    [0, 0, 0],         # unlabeled
    [110, 193, 228],   # sky (additional)
    [100, 100, 100],   # building (additional)
    [200, 200, 200]    # void
])


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Use MobileNetV2 encoder (same as training)
        model = smp.Unet(
            encoder_name='mobilenet_v2',
            encoder_weights=None,
            in_channels=3,
            classes=23,
            activation=None
        )
        
        if os.path.exists('best_model.pth'):
            state_dict = torch.load('best_model.pth', map_location=device)
            model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def colorize_mask(mask, palette=CITYSCAPES_PALETTE):
    """Convert mask to RGB using palette"""
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id in range(min(23, palette.shape[0])):
        class_mask = (mask == class_id)
        colored_mask[class_mask] = palette[class_id]
    
    return colored_mask


def predict_image(model, device, image_np, image_size=512):
    """Predict segmentation for an image"""
    # Prepare image
    original_shape = image_np.shape[:2]
    
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image_np)
    tensor_image = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(tensor_image)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize to original size
    pred_mask = cv2.resize(
        pred_mask.astype(np.uint8), 
        (original_shape[1], original_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    return pred_mask


def load_test_image(idx):
    """Load a test image from test set"""
    try:
        test_images_dir = '/home/m25csa010/M25CSA010_Major/Q2/MLDLOPs_2026_Major_Exam/CameraRGB'
        test_masks_dir = '/home/m25csa010/M25CSA010_Major/Q2/MLDLOPs_2026_Major_Exam/CameraMask'
        
        # Get all files sorted
        image_files = sorted(os.listdir(test_images_dir))
        mask_files = sorted(os.listdir(test_masks_dir))
        
        # Calculate test set index (test set starts at 80% of data)
        test_idx = int(0.8 * len(image_files)) + (idx % len(image_files[int(0.8 * len(image_files)):]))
        
        # Ensure within bounds
        test_idx = min(test_idx, len(image_files) - 1)
        
        # Load image and mask
        image_file = image_files[test_idx]
        mask_file = mask_files[test_idx]
        
        image = cv2.imread(os.path.join(test_images_dir, image_file))
        if image is None:
            raise ValueError(f"Could not load image: {image_file}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(test_masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_file}")
        
        return image, mask
    except Exception as e:
        raise Exception(f"Error loading test image {idx}: {str(e)}")


# Main app
st.title("🏙️ CityScape Image Segmentation")

page = st.sidebar.radio("Navigation", ["📊 Training Metrics", "🎯 Prediction"])

if page == "📊 Training Metrics":
    st.header("Training Metrics and Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Test Set Performance")
        
        # Load test metrics
        if os.path.exists('test_metrics.json'):
            with open('test_metrics.json', 'r') as f:
                test_metrics = json.load(f)
            
            st.metric("mIOU Score", f"{test_metrics['mIOU']:.4f}")
            st.metric("mDice Score", f"{test_metrics['mDice']:.4f}")
            
            # Check if metrics are above threshold
            if test_metrics['mIOU'] > 0.48 and test_metrics['mDice'] > 0.48:
                st.success("✅ Both metrics are above 0.48 threshold!")
            else:
                st.warning("⚠️ Some metrics are below 0.48 threshold!")
        else:
            st.warning("Test metrics file not found. Please run training first.")
    
    with col2:
        st.subheader("Training Curves")
    
    # Display training plot
    if os.path.exists('plots/training_metrics.png'):
        img = Image.open('plots/training_metrics.png')
        st.image(img, use_column_width=True, caption="Training Loss, mIOU, and mDice Curves")
    else:
        st.warning("Training metrics plot not found. Please run training first.")
    
    # Display detailed metrics
    st.subheader("Detailed Training History")
    
    if os.path.exists('plots/metrics.json'):
        with open('plots/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.line_chart(metrics['val_miou_scores'], y_label='mIOU')
        
        with col2:
            st.line_chart(metrics['val_mdice_scores'], y_label='mDice')


elif page == "🎯 Prediction":
    st.header("Image Segmentation Predictions")
    
    # Load model
    with st.spinner('Loading model...'):
        model, device = load_model()
    
    st.subheader("Upload Test Images")
    
    # Upload multiple images
    uploaded_files = st.file_uploader(
        "Choose images (up to 4)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key='image_uploader'
    )
    
    if uploaded_files:
        for idx, uploaded_file in enumerate(min(4, len(uploaded_files))):
            st.markdown("---")
            
            # Read and process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Ensure RGB
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            
            # Predict
            with st.spinner(f'Processing image {idx + 1}...'):
                pred_mask = predict_image(model, device, image_np)
                pred_mask_colored = colorize_mask(pred_mask)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_np, caption=f"Input Image {idx + 1}", use_column_width=True)
            
            with col2:
                st.image(pred_mask_colored, caption=f"Predicted Segmentation {idx + 1}", use_column_width=True)
    
    # Test set examples
    st.markdown("---")
    st.subheader("Load Test Set Examples")
    
    if st.button("Load 4 Random Test Images"):
        with st.spinner('Loading test images...'):
            test_cols = st.columns(4)
            
            for i in range(4):
                with test_cols[i]:
                    try:
                        image, mask = load_test_image(i)
                        pred_mask = predict_image(model, device, image)
                        pred_mask_colored = colorize_mask(pred_mask)
                        mask_colored = colorize_mask(mask)
                        
                        st.image(image, caption=f"Test Image {i+1}", use_column_width=True)
                        
                        st.write("Ground Truth:")
                        st.image(mask_colored, use_column_width=True)
                        
                        st.write("Prediction:")
                        st.image(pred_mask_colored, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image {i+1}: {str(e)}")
    
    # Legend
    st.markdown("---")
    st.subheader("Segmentation Classes")
    
    class_names = [
        "Road", "Sidewalk", "Building", "Wall", "Fence",
        "Pole", "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
        "Sky", "Person", "Rider", "Car", "Truck",
        "Bus", "Train", "Motorcycle", "Bicycle", "Unlabeled",
        "Sky (alt)", "Building (alt)", "Void"
    ]
    
    cols = st.columns(4)
    for idx, class_name in enumerate(class_names):
        with cols[idx % 4]:
            color = CITYSCAPES_PALETTE[min(idx, len(CITYSCAPES_PALETTE)-1)]
            color_hex = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            st.markdown(f"<div style='background-color: {color_hex}; padding: 10px; border-radius: 5px;'><p style='color: white; margin: 0;'>{class_name}</p></div>", unsafe_allow_html=True)
