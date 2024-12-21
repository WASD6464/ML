import torch
import torch.nn

def grad_penalty(critic, real, fake, device = 'cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_imgs = real * epsilon + fake * (1 - epsilon)

    # Calculate socres
    mixed_scores = critic(interpolated_imgs)

    grad = torch.autograd.grad(
        inputs=interpolated_imgs,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    grad = grad.view(grad.shape[0], -1)
    grad_norm = grad.norm(2, dim=1)
    grad_penalty = torch.mean((grad_norm - 1) ** 2)
    return grad_penalty

