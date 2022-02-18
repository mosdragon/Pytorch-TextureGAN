import math
import random

import torch
# from torch.autograd import Variable
import numpy as np

from utils import transforms as custom_transforms
from models import save_network, GramMatrix
from utils.visualize import vis_image, vis_patch


from logger import debug, info, warn, error

device0 = torch.device("cuda:0")

def rand_between(a, b):
    return a + torch.round(torch.rand(1) * (b - a))[0]


def gen_input(img, skg, ini_texture, ini_mask, xcenter=64, ycenter=64, size=40):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh

    w, h = img.size()[1:3]
    xstart = max(int(xcenter - size / 2), 0)
    ystart = max(int(ycenter - size / 2), 0)
    xend = min(int(xcenter + size / 2), w)
    yend = min(int(ycenter + size / 2), h)

    input_texture = ini_texture  # torch.ones(img.size())*(1)
    input_sketch = skg[0:1, :, :]  # L channel from skg
    input_mask = ini_mask  # torch.ones(input_sketch.size())*(-1)

    input_mask[:, xstart:xend, ystart:yend] = 1

    input_texture[:, xstart:xend, ystart:yend] = img[:, xstart:xend, ystart:yend].clone()

    # return torch.cat((input_sketch.cpu().float(), input_texture.float(), input_mask), 0)
    return torch.cat((input_sketch, input_texture, input_mask), 0)

def get_coor(index, size):
    index = int(index)
    #get original coordinate from flatten index for 3 dim size
    w,h = size
    
    return ((index%(w*h))/h, ((index%(w*h))%h))

def gen_input_rand(img, skg, seg, size_min=40, size_max=60, num_patch=1):
    # generate input skg with random patch from img
    # input img,skg [bsx3xwxh], xcenter,ycenter, size
    # output bsx5xwxh
    
    bs, c, w, h = img.size()
    results = torch.Tensor(bs, 5, w, h)
    texture_info = []

    # text_info.append([xcenter,ycenter,crop_size])
    seg = seg / torch.max(seg) #make sure it's 0/1
    
    seg[:,0:int(math.ceil(size_min/2)),:] = 0
    seg[:,:,0:int(math.ceil(size_min/2))] = 0
    seg[:,:,int(math.floor(h-size_min/2)):h] = 0
    seg[:,int(math.floor(w-size_min/2)):w,:] = 0
    
    counter = 0
    for i in range(bs):
        counter = 0
        ini_texture = torch.ones(img[0].size()) * (1)
        ini_mask = torch.ones((1, w, h)) * (-1)
        temp_info = []
        
        for j in range(num_patch):
            crop_size = int(rand_between(size_min, size_max))
            
            seg_index_size = seg[i,:,:].view(-1).size()[0]
            seg_index = torch.arange(0,seg_index_size)
            seg_one = seg_index[seg[i,:,:].view(-1)==1]
            if len(seg_one) != 0:
                seg_select_index = int(rand_between(0,seg_one.view(-1).size()[0]-1))
                x,y = get_coor(seg_one[seg_select_index],seg[i,:,:].size())
            else:
                x,y = (w/2, h/2)
            
            temp_info.append([x, y, crop_size])
            res = gen_input(img[i], skg[i], ini_texture, ini_mask, x, y, crop_size)

            ini_texture = res[1:4, :, :]

        texture_info.append(temp_info)
        results[i, :, :, :] = res
    return results, texture_info

def gen_local_patch(patch_size, batch_size, eroded_seg, seg, img):
    # generate local loss patch from eroded segmentation
    
    bs, c, w, h = img.size()
    texture_patch = img[:, :, 0:patch_size, 0:patch_size].clone()

    if patch_size != -1:
        eroded_seg[:,0,0:int(math.ceil(patch_size/2)),:] = 0
        eroded_seg[:,0,:,0:int(math.ceil(patch_size/2))] = 0
        eroded_seg[:,0,:,int(math.floor(h-patch_size/2)):h] = 0
        eroded_seg[:,0,int(math.floor(w-patch_size/2)):w,:] = 0

    for i_bs in range(bs):
                
        i_bs = int(i_bs)
        seg_index_size = eroded_seg[i_bs,0,:,:].view(-1).size()[0]
        seg_index = torch.arange(0,seg_index_size).to(device0)

        seg_one = seg_index[eroded_seg[i_bs,0,:,:].view(-1)==1]
        if len(seg_one) != 0:
            random_select = int(rand_between(0, len(seg_one)-1))        
            x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())

        else:
            x,y = (w/2, h/2)

        if patch_size == -1:
            xstart = 0
            ystart = 0
            xend = -1
            yend = -1

        else:
            xstart = int(x-patch_size/2)
            ystart = int(y-patch_size/2)
            xend = int(x+patch_size/2)
            yend = int(y+patch_size/2)

        k = 1
        while torch.sum(seg[i_bs,0,xstart:xend,ystart:yend]) < k*patch_size*patch_size:
                
            try:
                k = k*0.9
                if len(seg_one) != 0:
                    random_select = int(rand_between(0, len(seg_one)-1))
            
                    x,y = get_coor(seg_one[random_select], eroded_seg[i_bs,0,:,:].size())
            
                else:
                    x,y = (w/2, h/2)
                xstart = (int)(x-patch_size/2)
                ystart = (int)(y-patch_size/2)
                xend = (int)(x+patch_size/2)
                yend = (int)(y+patch_size/2)
            except:
                break
                
            
        texture_patch[i_bs,:,:,:] = img[i_bs, :, xstart:xend, ystart:yend]
        
    return texture_patch

def renormalize(img):
    """
    Renormalizes the input image to meet requirements for VGG-19 pretrained network
    """

    forward_norm = (torch.ones(img.data.size()) * 0.5).to(device0)
    # forward_norm = Variable(forward_norm.to(device0))
    img = (img * forward_norm) + forward_norm  # add previous norm
    # return img
    
    # mean = img.data.new(img.data.size())
    # std = img.data.new(img.data.size())
    
    mean = torch.zeros(img.size()).to(device0)
    std = torch.zeros(img.size()).to(device0)
    
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    img = img - mean
    img = img / std

    return img

    
def train(model, train_loader, val_loader, input_stack, target_img, target_texture,
          segment, label,label_local, extract_content, extract_style, loss_graph, vis, epoch, args):

    netG = model["netG"]
    netD = model["netD"]
    netD_local = model["netD_local"]
    criterion_gan = model["criterion_gan"]
    criterion_pixel_l = model["criterion_pixel_l"]
    criterion_pixel_ab = model["criterion_pixel_ab"]
    criterion_feat = model["criterion_feat"]
    criterion_style = model["criterion_style"]
    criterion_texturegan = model["criterion_texturegan"]
    real_label = model["real_label"]
    fake_label = model["fake_label"]
    optimizerD = model["optimizerD"]
    optimizerD_local = model["optimizerD_local"]
    optimizerG = model["optimizerG"]

    N_batches = len(train_loader)
    for i, data in enumerate(train_loader):
        info(f"Epoch: {epoch} - Batch {i+1}/{N_batches}")

        # Detach is apparently just creating new Variable with cut off reference to previous node, so shouldn't effect the original
        # But just in case, let's do G first so that detaching G during D update don't do anything weird
        ############################
        # (1) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        img, skg, seg, eroded_seg, txt = data  # LAB with negeative value

        # debug(f"Types: {img.dtype}  {skg.dtype}  {txt.dtype}")
        # debug(f"Shapes: {img.shape}  {skg.shape}  {txt.shape}")
        
        if random.random() < 0.5:
            txt = img
        
        # output img/skg/seg rgb between 0-1
        # output img/skg/seg lab between 0-100, -128-128
        if args.color_space == 'lab':
            img = custom_transforms.normalize_lab(img).float()
            skg = custom_transforms.normalize_lab(skg).float()
            txt = custom_transforms.normalize_lab(txt).float()
            seg = custom_transforms.normalize_seg(seg).float()
            eroded_seg = custom_transforms.normalize_seg(eroded_seg).float()
            # seg = custom_transforms.normalize_lab(seg).float()
        elif args.color_space == 'rgb':
            img = custom_transforms.normalize_rgb(img).float()
            skg = custom_transforms.normalize_rgb(skg).float()
            txt = custom_transforms.normalize_rgb(txt).float()
            # seg=custom_transforms.normalize_rgb(seg).float()
        

        if not args.use_segmentation_patch:
            seg.fill_(1)
         
        bs, w, h = seg.size()

        seg = seg.view(bs, 1, w, h)
        seg = torch.cat((seg, seg, seg), 1)
        eroded_seg = eroded_seg.view(bs, 1, w, h)

        
        temp = torch.ones(seg.size()) * (1 - seg)
        temp[:, 1, :, :] = 0  # torch.ones(seg[:,1,:,:].size())*(1-seg[:,1,:,:]).float()
        temp[:, 2, :, :] = 0  # torch.ones(seg[:,2,:,:].size())*(1-seg[:,2,:,:]).float()

        txt = txt * seg + temp
        
        if args.input_texture_patch == 'original_image':
            inp, _ = gen_input_rand(img, skg, eroded_seg[:, 0, :, :], args.patch_size_min, args.patch_size_max,
                                    args.num_input_texture_patch)
        elif args.input_texture_patch == 'dtd_texture':
            inp, _ = gen_input_rand(txt, skg, eroded_seg[:, 0, :, :], args.patch_size_min, args.patch_size_max,
                                    args.num_input_texture_patch)
        
        batch_size, _, _, _ = img.size()

        img = img.to(device0)
        skg = skg.to(device0)
        seg = seg.to(device0)
        eroded_seg = eroded_seg.to(device0)
        txt = txt.to(device0)
        inp = inp.to(device0)

        inputv = inp
        gtimgv = img
        txtv = txt

        
        inv_idx = torch.arange(txt.size(0)-1, -1, -1).long().to(device0)
        txt_inv = txt.index_select(0, inv_idx)

        assert torch.max(seg) <= 1
        assert torch.max(eroded_seg) <= 1

        inputv = inp
        gtimgv = img
        txtv = txt
        # txtv_inv = txt_inv
        
        outputG = netG(inputv)

        outputl, outputa, outputb = torch.chunk(outputG, 3, dim=1)

        gtl, gta, gtb = torch.chunk(gtimgv, 3, dim=1)
        txtl, txta, txtb = torch.chunk(txtv, 3, dim=1)
        # txtl_inv,txta_inv,txtb_inv = torch.chunk(txtv_inv,3,dim=1)

        outputab = torch.cat((outputa, outputb), 1)
        gtab = torch.cat((gta, gtb), 1)
        txtab = torch.cat((txta, txtb), 1)

        if args.color_space == 'lab':
            outputlll = (torch.cat((outputl, outputl, outputl), 1))
            gtlll = torch.cat((gtl, gtl, gtl), 1)
            txtlll = torch.cat((txtl, txtl, txtl), 1)
        elif args.color_space == 'rgb':
            outputlll = outputG  # (torch.cat((outputl,outputl,outputl),1))
            gtlll = gtimgv  # (torch.cat((targetl,targetl,targetl),1))
            txtlll = txtv
        if args.loss_texture == 'original_image':
            targetl = gtl
            targetab = gtab
            targetlll = gtlll
        else:
            # if args.loss_texture == 'texture_mask':
            # remove baskground dtd
            #     txtl = segv[:,0:1,:,:]*txtl
            #     txtab=segv[:,1:3,:,:]*txtab
            #     txtlll=segv*txtlll
            # elif args.loss_texture == 'texture_patch':

            targetl = txtl
            targetab = txtab
            targetlll = txtlll

        ################## Global Pixel ab Loss ############################
        err_pixel_ab = args.pixel_weight_ab * criterion_pixel_ab(outputab, targetab)

        ################## Global Feature Loss############################
        out_feat = extract_content(renormalize(outputlll))[0]
        gt_feat = extract_content(renormalize(gtlll))[0]
        err_feat = args.feature_weight * criterion_feat(out_feat, gt_feat.detach())

        ################## Global D Adversarial Loss ############################
        
        netD.zero_grad()
        
        #return outputl, txtl
        if args.color_space == 'lab':
            outputD = netD(outputl)
        elif args.color_space == 'rgb':
            outputD = netD(outputG)
        # D_G_z2 = outputD.data.mean()

        # label.resize_(outputD.data.size())
        # labelv = Variable(label.fill_(real_label))
        
        label = real_label
        labelv = torch.ones(outputD.size()).to(device0) * real_label

        # debug(f"outputD: {outputD.dtype}    {outputD.shape}")
        # debug(f"labelv: {labelv.dtype}    {labelv.shape}")
        err_gan = args.discriminator_weight * criterion_gan(outputD, labelv)
        err_pixel_l = 0
        ################## Global Pixel L Loss ############################
             
        err_pixel_l = args.global_pixel_weight_l * criterion_pixel_l(outputl, targetl)

        err_style = 0
        err_texturegan = 0

        if args.local_texture_size == -1:  # global, no loss patch            
            ################## Global Style Loss ############################
            output_style_feat = extract_style(outputlll)
            target_style_feat = extract_style(targetlll)
            gram = GramMatrix()

            for m in range(len(output_style_feat)):
                gram_y = gram(output_style_feat[m])
                gram_s = gram(target_style_feat[m])

                err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())
                                    
        else: # local loss patch
            patchsize = args.local_texture_size
            netD_local.zero_grad()
             
            for p in range(args.num_local_texture_patch):
                texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg,seg, outputlll)
                gt_texture_patch = gen_local_patch(patchsize, batch_size, eroded_seg,seg, targetlll)

                texture_patchl = gen_local_patch(patchsize, batch_size, eroded_seg, seg,outputl)
                gt_texture_patchl = gen_local_patch(patchsize, batch_size, eroded_seg,seg, targetl)

                ################## Local Style Loss ############################

                output_style_feat = extract_style(texture_patch)
                target_style_feat = extract_style(gt_texture_patch)

                gram = GramMatrix()


                for m in range(len(output_style_feat)):
                    gram_y = gram(output_style_feat[m])
                    gram_s = gram(target_style_feat[m])

                    err_style += args.style_weight * criterion_style(gram_y, gram_s.detach())

                ################## Local Pixel L Loss ############################

                err_pixel_l += args.local_pixel_weight_l * criterion_pixel_l(texture_patchl, gt_texture_patchl)
            
            
                ################## Local D Loss ############################
                outputD_local = netD_local(torch.cat((texture_patchl, gt_texture_patchl),1))

                # label_local.resize_(outputD_local.data.size())
                labelv_local = (torch.ones(outputD_local.size()) * real_label).to(device0)


                # debug(f"Compute Texturegan err: {outputD_local.shape} --> {criterion_texturegan(outputD_local, labelv_local)}")
                err_texturegan += args.discriminator_local_weight * criterion_texturegan(outputD_local, labelv_local)
            
            loss_graph["gdl"].append(err_texturegan.item())
        
        ####################################
        # print(f"""Errors:
        #     err_pixel_l: {err_pixel_l}
        #     err_pixel_ab: {err_pixel_ab}
        #     err_gan: {err_gan}
        #     err_feat: {err_feat}
        #     err_style: {err_style}
        #     err_texturegan: {err_texturegan}
        # """)

        err_G = err_pixel_l + err_pixel_ab + err_gan + err_feat + err_style + err_texturegan
        err_G.backward(retain_graph=True)

        optimizerG.step()

        loss_graph["g"].append(err_G.item())
        loss_graph["gpl"].append(err_pixel_l.item())
        loss_graph["gpab"].append(err_pixel_ab.item())
        loss_graph["gd"].append(err_gan.item())
        loss_graph["gf"].append(err_feat.item())
        loss_graph["gs"].append(err_style.item())

        info('G:', err_G.item())

        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        
        netD.zero_grad()

        # labelv = label
        if args.color_space == 'lab':
            outputD = netD(gtl)
        elif args.color_space == 'rgb':
            outputD = netD(gtimgv)

        # label.resize_(outputD.data.size())
        labelv = (torch.ones(outputD.size()) * real_label).to(device0)

        errD_real = criterion_gan(outputD, labelv)
        # debug(f"outputD: {outputD.shape} {outputD.dtype}    errD_real: {errD_real.shape} {errD_real.dtype}")
        errD_real.backward()

        score = torch.ones(batch_size)
        _, cd, wd, hd = outputD.size()
        D_output_size = cd * wd * hd

        clamped_output_D = outputD.clamp(0, 1)
        clamped_output_D = torch.round(clamped_output_D)
        for acc_i in range(batch_size):
            score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

        real_acc = torch.mean(score)

        if args.color_space == 'lab':
            outputD = netD(outputl.detach())
        elif args.color_space == 'rgb':
            outputD = netD(outputG.detach())
        
        # label.resize_(outputD.data.size())
        labelv = (torch.ones(outputD.size()) * fake_label).to(device0)

        errD_fake = criterion_gan(outputD, labelv)
        # debug(f"outputD: {outputD.shape} {outputD.dtype}    errD_fake: {errD_fake.shape} {errD_fake.dtype}")
        errD_fake.backward()
        
        score = torch.ones(batch_size)
        _, cd, wd, hd = outputD.size()
        D_output_size = cd * wd * hd

        clamped_output_D = outputD.clamp(0, 1)
        clamped_output_D = torch.round(clamped_output_D)
        for acc_i in range(batch_size):
            score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

        fake_acc = torch.mean(1 - score)

        D_acc = (real_acc + fake_acc) / 2

        if D_acc.item() < args.threshold_D_max:
            # D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            loss_graph["d"].append(errD.item())
            optimizerD.step()
        else:
            loss_graph["d"].append(0)

        info('D:  ', 'real_acc  ', "%.2f  " % real_acc.item(), 'fake_acc  ', "%.2f  " % fake_acc.item(), 'D_acc  ', D_acc.item())

        ############################
        # (2) Update D local network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        
        if args.local_texture_size != -1:
            patchsize = args.local_texture_size
            x1 = int(rand_between(patchsize, args.image_size - patchsize))
            y1 = int(rand_between(patchsize, args.image_size - patchsize))

            x2 = int(rand_between(patchsize, args.image_size - patchsize))
            y2 = int(rand_between(patchsize, args.image_size - patchsize))

            netD_local.zero_grad()

            if args.color_space == 'lab':
                outputD_local = netD_local(torch.cat((targetl[:, :, x1:(x1 + patchsize), y1:(y1 + patchsize)],targetl[:, :, x2:(x2 + patchsize), y2:(y2 + patchsize)]),1))#netD_local(targetl)
            elif args.color_space == 'rgb':
                outputD = netD(gtimgv.detach())

            # label.resize_(outputD_local.data.size())
            labelv = (torch.ones(outputD_local.size()) * real_label).to(device0)

            errD_real_local = criterion_texturegan(outputD_local, labelv)
            errD_real_local.backward(retain_graph=True)

            score = torch.ones(batch_size)
            _, cd, wd, hd = outputD_local.size()
            D_output_size = cd * wd * hd

            clamped_output_D = outputD_local.clamp(0, 1)
            clamped_output_D = torch.round(clamped_output_D)
            for acc_i in range(batch_size):
                score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

            realreal_acc = torch.mean(score)

            x1 = int(rand_between(patchsize, args.image_size - patchsize))
            y1 = int(rand_between(patchsize, args.image_size - patchsize))

            x2 = int(rand_between(patchsize, args.image_size - patchsize))
            y2 = int(rand_between(patchsize, args.image_size - patchsize))


            if args.color_space == 'lab':
                #outputD_local = netD_local(torch.cat((txtl[:, :, x1:(x1 + patchsize), y1:(y1 + patchsize)],outputl[:, :, x2:(x2 + patchsize), y2:(y2 + patchsize)]),1))#outputD = netD(outputl.detach())
                outputD_local = netD_local(torch.cat((texture_patchl, gt_texture_patchl), 1).detach())
            elif args.color_space == 'rgb':
                outputD = netD(outputG.detach())
            
            # label.resize_(outputD_local.data.size())
            labelv = (torch.ones(outputD_local.size()) * fake_label).to(device0)

            errD_fake_local = criterion_gan(outputD_local, labelv)
            errD_fake_local.backward()
            score = torch.ones(batch_size)
            _, cd, wd, hd = outputD_local.size()
            D_output_size = cd * wd * hd

            clamped_output_D = outputD_local.clamp(0, 1)
            clamped_output_D = torch.round(clamped_output_D)
            for acc_i in range(batch_size):
                score[acc_i] = torch.sum(clamped_output_D[acc_i]) / D_output_size

            fakefake_acc = torch.mean(1 - score)

            D_acc = (realreal_acc +fakefake_acc) / 2

            if D_acc.item() < args.threshold_D_max:
                # D_G_z1 = output.data.mean()
                errD_local = errD_real_local + errD_fake_local
                loss_graph["dl"].append(errD_local.item())
                optimizerD_local.step()
            else:
                loss_graph["dl"].append(0)

            info('D local:  ', 'real_acc  ', "%.2f  " % realreal_acc.item(), 'fake_acc  ', "%.2f  " % fakefake_acc.item(), 'D_acc  ', D_acc.item())
            # if i % args.save_every == 0:
             #   save_network(netD_local, 'D_local', epoch, i, args)

        if i % args.save_every == 0:
            save_network(netG, 'G', epoch, i, args)
            save_network(netD, 'D', epoch, i, args)
            save_network(netD_local, 'D_local', epoch, i, args)
            
        # if i % args.visualize_every == 0:
        #     visualize_training(netG, val_loader, input_stack, target_img,target_texture, segment, vis, loss_graph, args)


