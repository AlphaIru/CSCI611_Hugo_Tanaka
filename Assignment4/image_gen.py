"""This is used for part_2"""
# coding: utf-8

# import resources
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models


def get_input(prompt, default):
    value = input(f"{prompt} [default: {default}]: ").strip()
    return value if value else default


# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# get the "features" portion of VGG19
vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

vgg = vgg.to(device)


def load_image(img_path, max_size=400, shape=None):
    """ Load in and transform an image, making sure the image
       is <= max_size pixels in the x-y dims. """
    if "http" in img_path:
        response = requests.get(img_path, timeout=30)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_path).convert("RGB")

    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # discard transparent alpha channel if present, add batch dimension
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def save_image(tensor, filename):
    image = im_convert(tensor)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close()


# print out VGG19 structure so you can see the names of various layers
# print(vgg)


def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',
            '28': 'conv5_1'
        }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor """
    _, depth, height, width = tensor.size()
    tensor = tensor.view(depth, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram


# This is for command inputs.
# -----------------------------
# inputs
# -----------------------------
content_path = get_input("Content image path", "SF.jpg")
style_path = get_input("Style image path", "goth.jpg")
max_size = int(get_input("Image max size", "256"))
steps = int(get_input("Number of steps", "1000"))
show_every = int(get_input("Save image every N steps", "200"))
print_every = int(get_input("Print progress every N steps", "10"))
alpha = float(get_input("Alpha (content weight)", "1"))
beta = float(get_input("Beta (style weight)", "1000000"))
lr = float(get_input("Learning rate", "0.003"))

# make output folder with settings in name
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"a{alpha}_b{beta}_steps{steps}_size{max_size}_{timestamp}"
output_dir = os.path.join("outputs", run_name)
os.makedirs(output_dir, exist_ok=True)

print("\nSaving results to:", output_dir)
print("Settings:")
print("  content:", content_path)
print("  style:", style_path)
print("  max_size:", max_size)
print("  steps:", steps)
print("  alpha:", alpha)
print("  beta:", beta)
print("  lr:", lr)
print("  print_every:", print_every)
print("  show_every:", show_every)

# save config
with open(os.path.join(output_dir, "config.txt"), "w", encoding="utf-8") as f:
    f.write(f"content: {content_path}\n")
    f.write(f"style: {style_path}\n")
    f.write(f"max_size: {max_size}\n")
    f.write(f"steps: {steps}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"beta: {beta}\n")
    f.write(f"lr: {lr}\n")
    f.write(f"print_every: {print_every}\n")
    f.write(f"show_every: {show_every}\n")
    f.write(f"device: {device}\n")


# load in content and style image
content = load_image(content_path, max_size=max_size).to(device)
# Resize style to match content, makes code easier
style = load_image(style_path, shape=content.shape[-2:]).to(device)

# save original images
save_image(content, os.path.join(output_dir, "content.png"))
save_image(style, os.path.join(output_dir, "style.png"))

# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer
# weighting earlier layers more will result in larger style artifacts
# notice we are excluding conv4_2, our content representation
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1
}

# iteration hyperparameters
optimizer = optim.Adam([target], lr=lr)

print("\nStarting style transfer...\n")

for ii in range(1, steps + 1):

    # get the features from your target image
    # then calculate the content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean(
        (target_features["conv4_2"] - content_features["conv4_2"]) ** 2
    )

    # the style loss
    # initialize the style loss to 0
    style_loss = 0

    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, depth, height, width = target_feature.shape

        # calculate the target gram matrix
        target_gram = gram_matrix(target_feature)

        # get the "style" style representation
        style_gram = style_grams[layer]

        # calculate the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean(
            (target_gram - style_gram) ** 2
        )

        # add to the style loss
        style_loss += layer_style_loss / (depth * height * width)

    # calculate the total loss
    total_loss = alpha * content_loss + beta * style_loss

    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii == 1:
        print("Initial style loss:", style_loss.item())

    # print progress
    if ii % print_every == 0:
        print(
            f"Step {ii}/{steps} | "
            f"Total: {total_loss.item():.4f} | "
            f"Content: {content_loss.item():.4f} | "
            f"Style: {style_loss.item():.4f}"
        )

    # save intermediate images
    if ii % show_every == 0:
        save_path = os.path.join(output_dir, f"step_{ii:04d}.png")
        save_image(target, save_path)
        print("Saved:", save_path)

# save final result
final_path = os.path.join(output_dir, "final.png")
save_image(target, final_path)
print("\nSaved final image to:", final_path)
print("All outputs are in:", output_dir)


