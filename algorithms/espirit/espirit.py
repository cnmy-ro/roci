import torch



def ifft2c(kspace):
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1))
    image = torch.fft.fftshift(image, dim=(-2, -1))
    return image


def espirit_calib(kspace, calib_size=24, kernel_size=6, nullspace_cutoff=0.02, eigenvalue_thresh=0.95):

    num_coils = kspace.shape[0]
    size = kspace.shape[1]  # Assuming square FOV
    
    # Build the Block-Hankel calib matrix
    calib_region = kspace[:, size//2-calib_size//2:size//2+calib_size//2, size//2-calib_size//2:size//2+calib_size//2]
    num_blocks = calib_size - kernel_size + 1
    A = torch.empty((num_blocks**2, kernel_size**2 * num_coils), dtype=torch.complex64) # Shape (num_blocks**2, kernel_size**2 * num_coils)
    for i in range(num_blocks):
        for j in range(num_blocks):
            calib_matrix_row = calib_region[:, i:i+kernel_size, j:j+kernel_size].reshape(num_coils, kernel_size**2).flatten()
            A[num_blocks*i + j, :] = calib_matrix_row.clone()
    
    # SVD to compute V
    U, S, VH = torch.linalg.svd(A, full_matrices=False)
    V = VH.T
    V = V / torch.norm(V, p=2, dim=[1], keepdim=True) # Normalize the vectors to unit norm

    # Extract V-parallel
    num_kernels = S[S/S.max() > nullspace_cutoff].shape[0]
    V_parallel = V[:, :num_kernels]

    # Build Gq matrices for each spatial location q
    kernels = torch.empty((num_kernels, num_coils, kernel_size, kernel_size), dtype=torch.complex64)
    for k in range(num_kernels):
        for c in range(num_coils):
            offset = c * kernel_size**2 
            kernels[k, c] = V_parallel[ offset : offset + kernel_size**2, k ].reshape(kernel_size, kernel_size).clone()
    pad_before, pad_after = size//2 - kernel_size//2, size//2 - kernel_size//2
    kernels = torch.nn.functional.pad(kernels, pad=(pad_before, pad_after, pad_before, pad_after, 0, 0, 0, 0), mode='constant', value=0)
    kernel_images = ifft2c(kernels)
    kernel_images = kernel_images * size**2 / kernel_size**2
    Gq = kernel_images.reshape(num_kernels, num_coils, size**2).permute((2,0,1)) # Shape (rows*cols, num_kernels, num_coils)

    # Eigendecompose GqH.Gq matrices
    GqHGq = torch.bmm(Gq.permute(0,2,1), Gq.conj()) # Shape (rows*cols, num_coils, num_coils)
    eigenvalues, eigenvectors = torch.linalg.eig(GqHGq)
    eigenvalues = eigenvalues.abs() / eigenvalues[:, 0:1].abs().max() # Hack to make eigenvalue range [0,1]
    eigenvectors = eigenvectors / torch.norm(eigenvectors, p=2, dim=[1], keepdim=True) # Normalize the vectors to unit norm
    # print(eigenvalues.min(), eigenvalues.max()) # TODO: All values are way smaller than 1. Should be 1 in foreground. Fix this.    

    # Select required CSM omponents. Actual CSM is the dominant component with eigenvalue=1.
    body_mask = eigenvalues[:,0].reshape(size,size) > eigenvalue_thresh
    csm_components, eigenvalue_maps = [], []
    for component in range(num_coils):
        csm_component = eigenvectors[:,:,component]   # Shape (rows*cols, num_coils)
        csm_component = csm_component.T.reshape(num_coils, size, size) * body_mask # Shape (num_coils, rows, cols)
        csm_components.append(csm_component)
        eigenvalue_maps.append(eigenvalues[:, component].reshape(size, size))
    return csm_components, eigenvalues