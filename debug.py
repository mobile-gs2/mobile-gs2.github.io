@torch.no_grad()
def residual_aware_dc_init(
    sh3, gaussians_xyz, cams_xyz,
    num_samples=512, alpha=2.0,
    cam_chunk=16, dir_chunk=64,
):
    device = sh3.device
    N = sh3.shape[0]
    M = cams_xyz.shape[0]

    dirs = fibonacci_sphere(num_samples, device)          # [S, 3]
    Y3 = eval_real_sh_3(dirs, origin_degree=3)            # [S, 16]

    # 预分配累计量
    num = torch.zeros(N, 3, device=device)                # 分子
    den = torch.zeros(N, 1, device=device)                # 分母

    for j in range(0, num_samples, dir_chunk):
        dirs_chunk = dirs[j:j+dir_chunk]                  # [s, 3]
        Y3_chunk = Y3[j:j+dir_chunk]                      # [s, 16]

        # f_chunk: [N, s, 3]
        f_chunk = torch.einsum("sm,nmc->nsc", Y3_chunk, sh3)

        # 计算 vis_w_chunk: [N, s]
        vis_w_chunk = torch.zeros(N, f_chunk.shape[1], device=device)

        for i in range(0, M, cam_chunk):
            cams_chunk = cams_xyz[i:i+cam_chunk]          # [c, 3]
            view_dirs = cams_chunk[None] - gaussians_xyz[:, None]  # [N, c, 3]
            view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-6)

            dots = torch.einsum("ncd,sd->ncs", view_dirs, dirs_chunk)  # [N, c, s]
            vis_w_chunk += dots.clamp(min=0).sum(dim=1)               # 累加 c

        vis_w_chunk = vis_w_chunk / M                                # mean over cams

        # 残差权重
        var_w = f_chunk.var(dim=2)                                   # [N, s]
        w = vis_w_chunk * (1.0 + alpha * var_w)                      # [N, s]

        # 累加 MC 期望
        num += (f_chunk * w[..., None]).sum(dim=1)                   # [N, 3]
        den += w.sum(dim=1, keepdim=True)                            # [N, 1]

    c0_rgb = num / (den + 1e-6)
    sh0 = c0_rgb / C0
    return sh0.unsqueeze(1)




        