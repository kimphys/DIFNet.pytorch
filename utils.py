import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(fusion, img_cat, img_1, img_2, put_type='mean', balance=0.01):

    loss1 = intensity_loss(fusion, img_1, img_2, put_type)
    loss2 = structure_loss(fusion, img_cat)

    return loss1 + balance * loss2

def create_putative(in1, in2, put_type):

    if put_type == 'mean':
        iput = (in1 + in2) / 2
    elif put_type == 'left':
        iput = in1
    elif put_type == 'right':
        iput = in2
    else:
        raise EOFError('No supported type!')

    return iput

def intensity_loss(fusion, img_1, img_2, put_type):

    inp = create_putative(img_1, img_2, put_type)

    # L2 norm
    loss = torch.norm(fusion - inp, 2)

    return loss

def gradient(x):

    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top 

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy

def create_structure(inputs):

    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W) # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:,0,:,:] = torch.sum(a_00,dim=1)
    structure[:,1,:,:] = torch.sum(a_01,dim=1)
    structure[:,2,:,:] = torch.sum(a_10,dim=1)
    structure[:,3,:,:] = torch.sum(a_11,dim=1)

    return structure

def structure_loss(fusion, img_cat):
    
    st_fusion = create_structure(fusion)
    st_input = create_structure(img_cat)

    # Frobenius norm
    loss = torch.norm(st_fusion - st_input)

    return loss


if __name__ == "__main__":
    fusion = torch.rand(5,3,4,4)
    img_1 = torch.rand(5,3,4,4)
    img_2 = torch.rand(5,1,4,4)
    img_cat = torch.cat([img_1,img_2],dim=1)

    loss = compute_loss(fusion, img_cat, img_1, img_2, put_type='mean', balance=0.01)

    print(loss)
