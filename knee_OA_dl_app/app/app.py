import streamlit as st
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.models as models
import torch.nn as nn
import timm


# def make_gradcam_heatmap(model, img_tensor, target_layer, pred_index=None):

#     def forward_hook(module, input, output):
#         global feature_maps
#         feature_maps = output.detach()

#     def backward_hook(module, grad_in, grad_out):
#         global gradients
#         gradients = grad_out[0].detach()

#     handle_forward = target_layer.register_forward_hook(forward_hook)
#     handle_backward = target_layer.register_backward_hook(backward_hook)

#     output = model(img_tensor.unsqueeze(0))
#     if pred_index is None:
#         pred_index = output.argmax(dim=1).item()
#     class_score = output[:, pred_index]
#     model.zero_grad()
#     class_score.backward()

#     handle_forward.remove()
#     handle_backward.remove()

#     pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
#     for i in range(pooled_gradients.shape[0]):
#         feature_maps[:, i, :, :] *= pooled_gradients[i]

#     heatmap = torch.mean(feature_maps, dim=1).squeeze()
#     heatmap = torch.clamp(heatmap, min=0)
#     heatmap /= torch.max(heatmap)

#     return heatmap.numpy()

def find_last_conv_layer(model):
    conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layer = module
    return conv_layer

def register_hooks(layer):
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]
        
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)
    
    return lambda: activations, lambda: gradients

def make_gradcam_heatmap(model, input_tensor):
    model.eval()
    input_tensor.requires_grad_(True)
    target_layer = find_last_conv_layer(model)
    get_activations, get_gradients = register_hooks(target_layer)
    
    # Forward pass
    scores = model(input_tensor)
    score_max_index = scores.argmax()
    score_max = scores[:, score_max_index]

    # Backward pass
    model.zero_grad()
    score_max.backward(retain_graph=True)

    # Get the gradients and features from the hooks
    gradients = get_gradients()
    activations = get_activations()
    
    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Weight the channels by corresponding gradients
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
        
    # Average the channels of the activations to get the heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.nn.functional.relu(heatmap)
    heatmap /= torch.max(heatmap)  # Normalize the heatmap
    
    return heatmap.cpu().detach().numpy()


# def save_and_display_gradcam(img, heatmap, alpha=0.4):
#     heatmap = np.uint8(255 * heatmap)
#     jet = cm.get_cmap("jet")
#     jet_colors = jet(np.arange(256))[:, :3]
#     jet_heatmap = jet_colors[heatmap]
#     jet_heatmap = Image.fromarray((jet_heatmap * 255).astype('uint8')).resize(img.size)
#     jet_heatmap = np.array(jet_heatmap) / 255
#     superimposed_img = jet_heatmap * alpha + np.array(img)
#     superimposed_img = Image.fromarray((superimposed_img * 255).astype('uint8'))

#     return superimposed_img

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Resize heatmap to match the size of the original image
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    heatmap = Image.fromarray(heatmap).resize(img.size, Image.LANCZOS)
    heatmap = np.array(heatmap)

    # Apply the heatmap to the original image
    heatmap = np.uint8(255 * plt.cm.jet(heatmap)[:, :, :3])  # Apply the colormap
    superimposed_img = heatmap * alpha + np.array(img) * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    # Convert back to PIL image
    superimposed_img = Image.fromarray(superimposed_img)
    
    return superimposed_img

# Streamlit page configuration
icon = Image.open("app/img/iitb_logo.png")
st.set_page_config(page_title="Knee OA Severity Analysis", page_icon=icon)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Model_names = ["Xception_LT", "Resnet_LT", "InceptionResnet_LT", "ViTB16_LT", "Xception_BAL", "Resnet_BAL", "InceptionResnet_BAL", "ViTB16_BAL"]
Model_names_to_timm_models = {
    "Xception_LT": "xception",
    "Resnet_LT": "resnet50",
    "InceptionResnet_LT": "inception_resnet_v2",
    "ViTB16_LT": "vit_base_patch16_224",
    "Xception_BAL": "xception",
    "Resnet_BAL": "resnet50",
    "InceptionResnet_BAL": "inception_resnet_v2",
    "ViTB16_BAL": "vit_base_patch16_224"
}
class_names = ["Healthy", "Doubtful", "Minimal", "Moderate", "Severe"]

models = {}
for model_name in Model_names:
    models[model_name] = timm.create_model(Model_names_to_timm_models[model_name], pretrained=False, num_classes=len(class_names))
    models[model_name].to(device)
    models[model_name].load_state_dict(torch.load(f"../checkpoints/{Model_names_to_timm_models[model_name]}_{model_name.split('_')[-1]}.pth"))
    
# UI Sidebar
with st.sidebar:
    st.image(icon)
    st.markdown("<h2 style='text-align: center; font-size: 24px;'>Team PADMA</h2>", unsafe_allow_html=True)
    st.caption("===DH 602 Course Project===")
    uploaded_file = st.file_uploader("Choose X-Ray image")

st.header("Knee Osteoarthritis Severity Analysis")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # Ensure tensor is the correct shape
    img_tensor = img_tensor.to(torch.float32).to(device)

    tabs = st.tabs([i for i in Model_names])
    for i, (model_name, model) in enumerate(models.items(), start=1):
        with tabs[i - 1]:
            model.eval()  # Set the model to evaluation mode
            try:
                y_pred = model(img_tensor)
                y_pred = torch.nn.functional.softmax(y_pred, dim=1)[0] * 100
                probability, class_index = torch.max(y_pred, 0)
                grade = class_names[class_index]
            except Exception as e:
                st.error(f"Error during model inference: {str(e)}")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(":camera: Input X-Ray Image")
                st.image(img, use_column_width=True)
                st.subheader(":white_check_mark: Model Prediction")
                st.metric(label="Severity", value=f"{grade} - {probability:.2f}%")
                st.metric(label="KL Grade", value=f"KL Grade - {class_index}")

            if y_pred is not None:
                # with col2:
                with col2:
                    # Assuming 'model.features' is the last conv layer, change as necessary
                    heatmap = make_gradcam_heatmap(model, img_tensor)
                    image = save_and_display_gradcam(img, heatmap)
                    st.subheader(":mag: Grad-CAM Image")
                    st.image(image, use_column_width=True)
                    st.subheader(":bar_chart: Analysis")
                    fig, ax = plt.subplots(figsize=(5, 2))
                    y_pred = y_pred.detach().cpu().numpy()
                    ax.barh(class_names, y_pred, height=0.55, align="center")
                    for i, (c, p) in enumerate(zip(class_names, y_pred)):
                        ax.text(p + 2, i - 0.2, f"{p:.2f}%")
                    ax.grid(axis="x")
                    ax.set_xlim([0, 120])
                    ax.set_xticks(range(0, 101, 20))
                    fig.tight_layout()
                    st.pyplot(fig)






# # Application body
# st.header("Knee Osteoarthritis Severity Analysis")
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# col1, col2 = st.columns(2)
# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert('RGB')
#     img_tensor = transform(img)

#     # Create tabs for each model's output
#     tabs = st.tabs([i for i in Model_names])
#     for i, (model_name, model) in enumerate(models.items(), start=1):
#         with tabs[i - 1]:
#             # model.eval()
#             y_pred = model(img_tensor.unsqueeze(0))
#             y_pred = torch.nn.functional.softmax(y_pred, dim=1)[0] * 100
#             probability, class_index = torch.max(y_pred, 0)
#             grade = class_names[class_index]

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.subheader(":camera: Input X-Ray Image")
#                 st.image(img, use_column_width=True)
#                 st.subheader(":white_check_mark: Prediction")
#                 st.metric(label="Severity", value=f"{grade} - {probability:.2f}%")
#                 st.metric(label="KL Grade", value=f"KL Grade - {class_index}")

#             if y_pred is not None:
#                 with col2:
#                     # Assuming 'model.features' is the last conv layer, change as necessary
#                     heatmap = make_gradcam_heatmap(model, img_tensor, model.features)
#                     image = save_and_display_gradcam(img, heatmap)
#                     st.subheader(":mag: Gradcam Image")
#                     st.image(image, use_column_width=True)
#                     st.subheader(":bar_chart: Analysis")
#                     fig, ax = plt.subplots(figsize=(5, 2))
#                     ax.barh(class_names, y_pred.numpy(), height=0.55, align="center")
#                     for i, (c, p) in enumerate(zip(class_names, y_pred.numpy())):
#                         ax.text(p + 2, i - 0.2, f"{p:.2f}%")
#                     ax.grid(axis="x")
#                     ax.set_xlim([0, 120])
#                     ax.set_xticks(range(0, 101, 20))
#                     fig.tight_layout()
#                     st.pyplot(fig)
