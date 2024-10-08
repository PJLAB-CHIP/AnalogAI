import torch

def FGSM(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    perturbed_image = perturbed_image.detach()
    perturbed_image.requires_grad = True
    return perturbed_image



def PGD(model, images, labels, loss_fn, epsilon, alpha, num_steps):
    # Set the model to evaluation mode
    # model.eval()
    # perturbed_images = images.clone().detach()
    # perturbed_images.requires_grad = True
    original_images = images.clone().detach()
    images = images + torch.randn_like(images) * epsilon
    images = torch.clamp(images, 0, 1)

    for i in range(num_steps):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
    
    return images