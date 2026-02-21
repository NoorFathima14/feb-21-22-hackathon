import torch
import torch.nn as nn

def predict_with_confidence(model, image_tensor, device):

    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item()


def fgsm_attack(model, image, label, epsilon, device):

    model.eval()

    image = image.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    image.requires_grad = True

    output = model(image)
    loss = nn.CrossEntropyLoss()(output, label)

    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()

    # Clamp to normalized range [-1, 1]
    perturbed_image = torch.clamp(perturbed_image, -1, 1)

    return perturbed_image.squeeze(0).detach()