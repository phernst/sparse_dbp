import os
from os.path import join as pjoin
from utils import DATA_DIRS

import ctl
import nibabel as nib
import numpy as np
import torch
from torch.nn.functional import grid_sample


# meshgrid : [3, ...]
# returns: [2, ...]
def project_onto_detector(meshgrid, pmat):
    mshape = meshgrid.shape
    meshgrid = meshgrid.view(3, -1).float()
    meshgrid = torch.cat([
        meshgrid,
        torch.ones(1, meshgrid.shape[-1], device=meshgrid.device)
    ], dim=0)
    hom_det = torch.mm(pmat, meshgrid)  # [3, ...]
    detector = hom_det[:2].clone()  # [2, ...]
    detector[0] /= hom_det[2]
    detector[1] /= hom_det[2]
    detector = detector.view(2, *mshape[1:])
    return detector[0], detector[1]


def calculate_normalization(a_lambda, meshgrid):
    return 1./torch.norm(meshgrid.permute(1, 2, 3, 0) - a_lambda, dim=3)


# image: [W, H]
# samples_x: [W, H]
# samples_y: [W, H]
def bilinear_interpolate(image, samples_x, samples_y):
    W, H = image.shape
    samples_x = samples_x.unsqueeze(0)
    samples_x = samples_x.unsqueeze(3)
    samples_y = samples_y.unsqueeze(0)
    samples_y = samples_y.unsqueeze(3)
    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0]/(W - 1))
    samples[:, :, :, 1] = (samples[:, :, :, 1]/(H - 1))
    samples = samples*2 - 1
    return grid_sample(
        image[None, None, ...],
        samples,
        align_corners=True)[0, 0]


class trapezoidal_enumerate:
    def __init__(self, l):
        assert isinstance(l, list)
        self.l = l
        self.weights = [1]*len(l)
        self.weights[0] = self.weights[-1] = 0.5

    def __iter__(self):
        return zip(self.weights, self.l)


# uv_grid: [u, v, 2]
# returns [u, v, 3]
def uv_to_raydir(uv_grid, projection_matrix):
    # make coordinates homogeneous
    uvw_grid = torch.cat([
        uv_grid,
        torch.ones(*uv_grid.shape[:2], 1, dtype=torch.double)
    ], dim=-1)

    M = projection_matrix[:, :3]
    q, r = torch.qr(M.transpose(0, 1))
    r_sign = r.diag().prod().sign()

    back_sub = torch.triangular_solve(
        uvw_grid.unsqueeze(-1).cuda(),
        r,
        transpose=True)[0]  # (u, v, 3, 1)
    raydirs = r_sign*torch.matmul(q, back_sub)[..., 0]
    norm_raydirs = raydirs/torch.norm(raydirs, dim=-1, keepdim=True)
    return norm_raydirs


# x, y: (*, n) or broadcastable
# returns (*, 1), i.e. inner product along last axis
def bdot(x, y):
    return torch.sum(x*y, dim=-1, keepdim=True)


# source_position: numpy array, 3d vector
# ray direction: [*, 3], unit vectors
# returns [*, 3]
def point_of_interest(source_position, ray_direction):
    # x0 = np.array([0.0]*3)
    # n = np.array([0.0, 0.0, 1.0])
    scaling = bdot(source_position[:2], ray_direction[..., :2]) /\
        (1. - ray_direction[..., 2:]*ray_direction[..., 2:])
    return source_position - scaling*ray_direction


# only works for circular trajectories
def compute_delta_lambda(pmat0, pmat1, source_to_iso):
    source_pos_0 = torch.from_numpy(np.array(pmat0.source_position())[:, 0])
    source_pos_1 = torch.from_numpy(np.array(pmat1.source_position())[:, 0])
    source_pos_0 = source_pos_0.cuda()
    source_pos_1 = source_pos_1.cuda()
    return torch.acos(source_pos_0.dot(source_pos_1)/(source_to_iso**2))


# source_position: 3d vector
# delta_lambda in rad
# eps in [0,1]
def compute_source_position(source_position, delta_lambda, eps):
    x, y, z = source_position
    theta = torch.tensor([delta_lambda*eps], device='cuda')
    return torch.tensor([
        x*theta.cos() - y*theta.sin(),
        x*theta.sin() + y*theta.cos(),
        z
    ], device='cuda')


# eps in rad
def compute_source_position_absolute(source_position, eps):
    x, y, z = source_position
    theta = torch.tensor([eps], device='cuda')
    return torch.tensor([
        x*theta.cos() - y*theta.sin(),
        x*theta.sin() + y*theta.cos(),
        z
    ], device='cuda')


# equation 24
# poi_grid: [3, u, v]
def compute_g_approx(g0, g1, poi_grid, pmat0, pmat1, eps):
    pmat0_t = torch.from_numpy(np.array(pmat0)).float().cuda()
    pmat1_t = torch.from_numpy(np.array(pmat1)).float().cuda()
    g0_coords = project_onto_detector(poi_grid, pmat0_t)
    g1_coords = project_onto_detector(poi_grid, pmat1_t)
    g0_interp = bilinear_interpolate(g0, g0_coords[0], g0_coords[1])
    g1_interp = bilinear_interpolate(g1, g1_coords[0], g1_coords[1])
    return (1 - eps)*g0_interp + eps*g1_interp


def compute_single_g_approx(g, poi_grid, pmat):
    pmat_t = torch.from_numpy(np.array(pmat)).float().cuda()
    g_coords = project_onto_detector(poi_grid, pmat_t)
    g_interp = bilinear_interpolate(g, g_coords[0], g_coords[1])
    return g_interp


def compute_g_approx_absolute(g0, g1, poi_grid,
                              pmat0, pmat1, eps, delta_lambda):
    return compute_g_approx(g0, g1, poi_grid, pmat0, pmat1, eps/delta_lambda)


def vector_length(vec):
    return vec.dot(vec).sqrt()


# equation 25
# uv_grid: [u, v, 2], e.g. stack(meshgrid(arange(1024), arange(1024)), dim=-1).double()+.5
# eps in [0,1]
def compute_g_prime(g0, gm, gp, pmat0, pmatm, pmatp, uv_grid, eps):
    pmat0_mat = torch.from_numpy(np.array(pmat0)).cuda()
    dir_grid = uv_to_raydir(uv_grid, pmat0_mat)
    source_pos_0 = torch.from_numpy(np.array(pmat0.source_position())[:, 0]).cuda()
    source_to_iso = vector_length(source_pos_0)
    delta_lambda = compute_delta_lambda(pmat0, pmatp, source_to_iso)
    poip = point_of_interest(
        compute_source_position(source_pos_0, delta_lambda, eps), dir_grid)
    poim = point_of_interest(
        compute_source_position(source_pos_0, -delta_lambda, eps), dir_grid)
    g_approx_p = compute_g_approx(g0, gp, poip.permute(2, 0, 1),
                                  pmat0, pmatp, eps)
    g_approx_m = compute_g_approx(g0, gm, poim.permute(2, 0, 1),
                                  pmat0, pmatm, eps)
    return (g_approx_p - g_approx_m)/(2*eps*delta_lambda)


# eps in rad
def compute_g_prime_absolute(g0, gm, gp, pmat0, pmatm, pmatp, uv_grid, eps):
    pmat0_mat = torch.from_numpy(np.array(pmat0)).cuda()
    dir_grid = uv_to_raydir(uv_grid, pmat0_mat) # [u, v, 3]
    source_pos_0 = torch.from_numpy(np.array(pmat0.source_position())[:, 0]).cuda()
    source_to_iso = vector_length(source_pos_0)
    delta_lambda = compute_delta_lambda(pmat0, pmatp, source_to_iso)
    poip = point_of_interest(
        compute_source_position_absolute(source_pos_0, eps), dir_grid)
    poim = point_of_interest(
        compute_source_position_absolute(source_pos_0, -eps), dir_grid)
    g_approx_p = compute_g_approx_absolute(g0, gp, poip.permute(2, 0, 1),
                                           pmat0, pmatp, eps, delta_lambda)
    g_approx_m = compute_g_approx_absolute(g0, gm, poim.permute(2, 0, 1),
                                           pmat0, pmatm, eps, delta_lambda)
    return (g_approx_p - g_approx_m)/(2*eps)


# vectors in wcs: (x, y, z)
# vectors in b: (t, z)
# b_meshgrid: (t_grid, z_grid), i.e. created with indexing='ij'
def approximal_b_sagittal(s, dlambda, gfs_and_pmats, b_meshgrid):
    b_hat = torch.zeros(b_meshgrid[0].shape, device='cuda')
    x_meshgrid = torch.stack([
        -s*torch.ones(1, *b_meshgrid[0].shape, device='cuda'),
        b_meshgrid[0][None, ...].cuda(),
        b_meshgrid[1][None, ...].cuda(),
    ])
    for weight, (gf, pmat) in trapezoidal_enumerate(gfs_and_pmats):
        a_lambda = torch.from_numpy(np.array(pmat.source_position())[:, 0]).cuda()
        normalization = calculate_normalization(a_lambda, x_meshgrid)

        gf_t = gf
        pmat_t = torch.from_numpy(np.array(pmat)).float().cuda()
        u_samples, v_samples = project_onto_detector(x_meshgrid, pmat_t)
        u_samples = u_samples[0]  # only one x-coordinate == s
        v_samples = v_samples[0]  # only one x-coordinate == s
        evaluated_gf = bilinear_interpolate(gf_t, u_samples, v_samples)

        b_hat = b_hat + weight*normalization[0]*evaluated_gf
    b_hat = b_hat*dlambda
    return b_hat


# vectors in wcs: (x, y, z)
# vectors in b: (t, z)
# b_meshgrid: (t_grid, z_grid), i.e. created with indexing='ij'
def approximal_b_coronal(s, dlambda, gfs_and_pmats, b_meshgrid):
    b_hat = torch.zeros_like(b_meshgrid[0])
    x_meshgrid = torch.stack([
        b_meshgrid[0][:, None, ...],
        s*torch.ones_like(b_meshgrid[0][:, None, ...]),
        b_meshgrid[1][:, None, ...],
    ])
    for weight, (gf, pmat) in trapezoidal_enumerate(gfs_and_pmats):
        a_lambda = torch.from_numpy(np.array(pmat.source_position())[:, 0])
        a_lambda = a_lambda.to(b_hat.device)
        normalization = calculate_normalization(a_lambda, x_meshgrid)

        gf_t = gf
        pmat_t = torch.from_numpy(np.array(pmat)).float().to(b_hat.device)
        u_samples, v_samples = project_onto_detector(x_meshgrid, pmat_t)
        u_samples = u_samples[:, 0]  # only one y-coordinate == -s
        v_samples = v_samples[:, 0]  # only one y-coordinate == -s
        evaluated_gf = bilinear_interpolate(gf_t, u_samples, v_samples)

        b_hat = b_hat + weight*normalization[1]*evaluated_gf
    b_hat = b_hat*dlambda
    return b_hat


def test_approximal_b_sagittal(filename,
                               data_dir,
                               num_views,
                               sdd,
                               sid,
                               num_det_pixels,
                               det_pix_dim,
                               out_dir):
    nib_volume = nib.load(pjoin(data_dir, filename))
    spacings = np.array([
        float(f) for f in nib_volume.header['pixdim'][1:4]] + [1.])
    nib_volume = nib_volume.get_fdata()
    volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    volume.set_voxel_size(tuple(spacings))

    angle_diff = 360/num_views
    dlambda = np.deg2rad(angle_diff)

    system = ctl.CTSystem()
    system.add_component(ctl.FlatPanelDetector(
        (num_det_pixels, num_det_pixels),
        (det_pix_dim, det_pix_dim)))
    system.add_component(ctl.TubularGantry(sdd, sid))
    system.add_component(ctl.XrayTube())

    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projection_matrices = ctl.GeometryEncoder.encode_full_geometry(setup)
    projection_matrices = [p[0] for p in projection_matrices]

    print("create the projections")
    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume).numpy()
    projections = torch.from_numpy(projections).float().cuda()

    print("calculate the derivatives")
    der_grid = torch.stack(
        torch.meshgrid(torch.arange(num_det_pixels),
                       torch.arange(num_det_pixels)), dim=-1) + .5
    der_grid = der_grid.double()
    projections_der = [
        compute_g_prime_absolute(
            projections[i][0],
            projections[(i - 1) % num_views][0],
            projections[(i + 1) % num_views][0],
            projection_matrices[i],
            projection_matrices[(i - 1) % num_views],
            projection_matrices[(i + 1) % num_views],
            der_grid,
            2e-3).transpose(0, 1).cpu().numpy()
        for i in range(num_views)
    ]
    gfs = projections_der
    gfs = [torch.from_numpy(gf).cuda() for gf in gfs]
    gfs_and_pmats = list(zip(gfs, projection_matrices))

    b_meshgrid = torch.meshgrid(
        (torch.arange(nib_volume.shape[0], dtype=torch.float, device='cuda')-nib_volume.shape[0]/2)*spacings[0],
        (torch.arange(nib_volume.shape[2], dtype=torch.float, device='cuda')-nib_volume.shape[2]/2)*spacings[2],
    )

    hilbert_volume = torch.zeros(*nib_volume.shape, 2, dtype=torch.float, device='cuda')
    for x_idx in range(nib_volume.shape[0]):
        print(x_idx)
        s = (nib_volume.shape[0]/2 - x_idx)*spacings[0]
        angle = np.rad2deg(np.arccos(s/sid))
        idx_angle = int(angle//angle_diff)
        b_hat = approximal_b_sagittal(s, dlambda, gfs_and_pmats[idx_angle:-idx_angle+1], b_meshgrid)
        b_hat_rev = approximal_b_sagittal(
            s,
            dlambda,
            gfs_and_pmats[-idx_angle:] + gfs_and_pmats[:idx_angle+1],
            b_meshgrid
        )
        hilbert_volume[x_idx, :, :, 0] = b_hat
        hilbert_volume[x_idx, :, :, 1] = b_hat_rev

    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(xyz=2)  # mm
    img = nib.Nifti1Image(hilbert_volume.cpu().numpy(), np.diag(spacings), header=hdr)
    nib.save(img, pjoin(out_dir, f'{filename[:-7]}_sagittal.nii.gz'))


def test_approximal_b_coronal(filename, data_dir, num_views, sdd, sid, num_det_pixels, det_pix_dim, out_dir):
    nib_volume = nib.load(pjoin(data_dir, filename))
    spacings = np.array([float(f) for f in nib_volume.header['pixdim'][1:4]] + [1.])
    nib_volume = nib_volume.get_fdata()
    volume = ctl.VoxelVolumeF.from_numpy(nib_volume.transpose())
    volume.set_voxel_size(tuple(spacings))

    angle_diff = 360/num_views
    dlambda = np.deg2rad(angle_diff)

    system = ctl.CTSystem()
    system.add_component(ctl.FlatPanelDetector(
        (num_det_pixels, num_det_pixels),
        (det_pix_dim, det_pix_dim)))
    system.add_component(ctl.TubularGantry(sdd, sid))
    system.add_component(ctl.XrayTube())

    setup = ctl.AcquisitionSetup(system, num_views)
    setup.apply_preparation_protocol(ctl.protocols.AxialScanTrajectory())

    projection_matrices = ctl.GeometryEncoder.encode_full_geometry(setup)
    projection_matrices = [p[0] for p in projection_matrices]

    print("create the projections")
    projector = ctl.ocl.RayCasterProjector()
    projections = projector.configure_and_project(setup, volume).numpy()
    projections = torch.from_numpy(projections).float().cuda()

    print("calculate the derivatives")
    der_grid = torch.stack(
        torch.meshgrid(torch.arange(num_det_pixels),
                       torch.arange(num_det_pixels)), dim=-1)+.5
    der_grid = der_grid.double()
    projections_der = [
        compute_g_prime_absolute(
            projections[i][0],
            projections[(i - 1) % num_views][0],
            projections[(i + 1) % num_views][0],
            projection_matrices[i],
            projection_matrices[(i - 1) % num_views],
            projection_matrices[(i + 1) % num_views],
            der_grid,
            2e-3).transpose(0, 1).cpu().numpy()
        for i in range(num_views)
    ]
    gfs = projections_der
    gfs = [torch.from_numpy(gf).cuda() for gf in gfs]
    gfs_and_pmats = list(zip(gfs, projection_matrices))

    b_meshgrid = torch.meshgrid(
        (torch.arange(nib_volume.shape[1], dtype=torch.float, device=gfs[0].device)-nib_volume.shape[1]/2)*spacings[1],
        (torch.arange(nib_volume.shape[2], dtype=torch.float, device=gfs[0].device)-nib_volume.shape[2]/2)*spacings[2],
    )

    hilbert_volume = torch.zeros(*nib_volume.shape, 2, dtype=torch.float, device=gfs[0].device)
    for y_idx in range(nib_volume.shape[1]):
        print(y_idx)
        s = (y_idx - nib_volume.shape[1]/2)*spacings[1]
        angle = np.rad2deg(np.arcsin(s/sid))
        idx_angle = int(angle//angle_diff)
        idx_180 = int((180 - angle)//angle_diff)
        if angle >= 0:
            b_hat = approximal_b_coronal(s, dlambda, gfs_and_pmats[idx_angle:idx_180+1], b_meshgrid)
            b_hat_rev = approximal_b_coronal(
                s, dlambda,
                gfs_and_pmats[idx_180:] + gfs_and_pmats[:idx_angle],
                b_meshgrid,
            )
        else:
            b_hat = approximal_b_coronal(
                s, dlambda,
                gfs_and_pmats[idx_angle:] + gfs_and_pmats[:idx_180+1],
                b_meshgrid,
            )
            b_hat_rev = approximal_b_coronal(
                s, dlambda,
                gfs_and_pmats[idx_180:idx_angle],
                b_meshgrid,
            )
        hilbert_volume[:, y_idx, :, 0] = b_hat
        hilbert_volume[:, y_idx, :, 1] = b_hat_rev

    hdr = nib.Nifti1Header()
    hdr.set_xyzt_units(xyz=2)  # mm
    img = nib.Nifti1Image(
        hilbert_volume.cpu().numpy(),
        np.diag(spacings), header=hdr)
    nib.save(img, pjoin(out_dir, f'{filename[:-7]}_coronal.nii.gz'))


def directory_to_hilbert_volumes(path: str):
    num_views = 36
    sdd = 1000.
    sid = 750.
    num_det_pixels = 1024
    det_pix_dim = 1.
    data_dir = os.path.normpath(path)
    out_dir = f'views_{num_views}'

    processing_args = (
        data_dir,
        num_views,
        sdd,
        sid,
        num_det_pixels,
        det_pix_dim,
        out_dir,
    )

    for filename in sorted(os.listdir(data_dir))[-1:]:
        print(filename)
        test_approximal_b_coronal(filename, *processing_args)
        test_approximal_b_sagittal(filename, *processing_args)


if __name__ == "__main__":
    directory_to_hilbert_volumes(DATA_DIRS['datasets'])
