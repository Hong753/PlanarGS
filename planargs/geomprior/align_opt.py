import torch
from planargs.common_utils.loss_utils import l2_loss

def OptimizeGroupDepth(source, target, weight, prep, device="cuda"):
    """
    Optimize scale and shift to minimize the discrepancy between the source and target depth maps.
    ------------------------------------------------------
    Arguments
    source: list[torch.tensor(h,w)]
    target: list[torch.tensor(h,w)]
    ------------------------------------------------------
    Returns
    refined_source_list: list[torch.tensor(h,w)]
    loss: float
    """
    refined_source = []
    source_masked = []
    target_masked = []
    weight_masked = []
    for idx in range(len(source)):
        source_img = source[idx]
        target_img = target[idx]
        weight_img = weight[idx]
        mask_img1 = target_img > 0
        mask_img2 = source_img > 0
        mask_img = torch.logical_and(mask_img1, mask_img2)

        # Prune some depths considered "outlier" 
        with torch.no_grad():
            if target_img.max() == 0:
                mask_img = torch.zeros_like(target_img, dtype=bool)
            else:
                target_img_depth_sorted = target_img[target_img>1e-7].sort().values
                min_prune_threshold = target_img_depth_sorted[int(target_img_depth_sorted.numel() * prep.prune_ratio)]
                max_prune_threshold = target_img_depth_sorted[int(target_img_depth_sorted.numel() * (1.0 - prep.prune_ratio))]

                mask_img2 = target_img > min_prune_threshold
                mask_img3 = target_img < max_prune_threshold
                mask_img = torch.logical_and(torch.logical_and(mask_img, mask_img2), mask_img3)

        source_masked.append(source_img[mask_img])
        target_masked.append(target_img[mask_img])
        weight_masked.append(weight_img[mask_img])

    scale = torch.ones(1).to(device).requires_grad_(True)
    shift = (torch.ones(1) * 0.5).to(device).requires_grad_(True)

    optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
    loss = torch.ones(1).to(device) * 1e5

    iteration = 1
    loss_prev = 1e6
    loss_ema = 0.0

    stacked_source = torch.cat(source_masked, dim=0).requires_grad_(True)
    stacked_target = torch.cat(target_masked, dim=0).requires_grad_(True)
    stacked_weight = torch.cat(weight_masked, dim=0).requires_grad_(False)

    while abs(loss_ema - loss_prev) > prep.align_loss:
        stacked_source_hat = scale * stacked_source + shift
        loss = l2_loss(stacked_source_hat, stacked_target, stacked_weight)

        loss_hinge1 = loss_hinge2 =  0.0
        if (stacked_source_hat <= 0.0).any():
            loss_hinge1 = 2.0*((stacked_source_hat[stacked_source_hat <= 0.0])**2).mean()
        
        loss = loss + loss_hinge1 + loss_hinge2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        iteration+=1
        if iteration % 10000 == 0:
            print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
            loss_prev = loss.item()
        loss_ema = loss.item() * 0.2 + loss_ema * 0.8

    print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
    
    loss = loss.item()
    print(f"loss ={loss:10.5f}")

    with torch.no_grad():
        for idx in range(len(source)):
            refined_source_img = scale * source[idx] + shift
            refined_source.append(refined_source_img)
    torch.cuda.empty_cache()
    param = [scale.item(), shift.item()]

    return refined_source, param, loss


