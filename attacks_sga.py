import torch
import torch.nn as nn
import numpy as np



def cal_loss(loader, model, delta, beta, loss_function):
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data[0].cuda()  #mytest
            outputs_ori = model(x_val.cuda())
            _, target_label = torch.max(outputs_ori, 1)
            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)
            if loss_function:
                loss = torch.mean(loss_fn(outputs, target_label))
            else:
                loss = torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss_total = loss_total + loss
    loss_total = loss_total / (i + 1)
    return loss_total


def cal_loss_targeted(loader, model, delta, beta, loss_function, target_class):
    loss_total = 0
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    delta = delta.cuda()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x_val = data[0].cuda()
            y_val = data[1].cuda()
            outputs_ori = model(x_val.cuda())
            _, target_label = torch.max(outputs_ori, 1)
            perturbed = torch.clamp((x_val + delta), 0, 1)
            outputs = model(perturbed)

            y_target = torch.ones(size=y_val.shape, dtype=y_val.dtype) * target_class

            if loss_function:
                loss = -torch.mean(loss_fn(outputs, y_target.cuda()))
            else:
                loss = -torch.mean(outputs.gather(1, (y_target.cuda()).unsqueeze(1)).squeeze(1))
            loss_total = loss_total + loss
    loss_total = loss_total / (i + 1)
    return loss_total



def uap_sga_test(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
            uap_init=None, batch_size=None, minibatch = 10, loader_eval=None, dir_uap=None, center_crop=224, iter=4, Momentum=0,img_num=10000):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    minibatch   minibatch for SGA
    iter        iteration number (K) in the inner iteration
    center_crop image size
    Momentum    momentum item (default is false)
    
    log output
    batch_size  batch size 
    loader_eval evaluation dataloader
    dir_uap     save patch
    img_num     total image num
    '''
    target_class = 174
    model.eval()
    DEVICE = torch.device("cuda:0")
    np.random.seed(0)
    if uap_init is None:
        batch_delta = torch.zeros(batch_size, 3, center_crop, center_crop)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    # loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss


    batch_delta.requires_grad_()
    v = 0
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay
        #for i, data in enumerate(loader):
        #    x_val = data
        for i, (x_val, y_val) in enumerate(loader):
            with torch.no_grad():
                outputs_ori = model(x_val.cuda())
                _, target_label = torch.max(outputs_ori, 1)
                y_target = torch.ones(size=y_val.shape, dtype=y_val.dtype) * target_class

            num = x_val.shape[0]
            k = iter
            noise_inner_all = torch.zeros(k * num//minibatch, 3, center_crop, center_crop)
            delta_inner = delta.data
            for j in range(k * num//minibatch):
                label = np.random.choice(num, minibatch, replace=False)
                if j > 0 or i > 0 or epoch>0:
                    batch_delta.grad.data.zero_()
                batch_delta.data = delta_inner.unsqueeze(0).repeat([minibatch, 1, 1, 1])
                perturbed = torch.clamp((x_val[label] + batch_delta).cuda(), 0, 1)
                outputs = model(perturbed)
                #loss function value
                if loss_function:
                    loss = -clamped_loss(outputs, y_target[label].cuda())
                else:
                    loss = torch.mean(outputs.gather(1, (y_target[label].cuda()).unsqueeze(1)).squeeze(1))

                loss.backward()
                grad_inner = batch_delta.grad.data.mean(dim=0)
                delta_inner = delta_inner + grad_inner.sign() * eps_step
                delta_inner = torch.clamp(delta_inner, -eps, eps)
                noise_inner_all[j, :, :, :] = grad_inner
                batch_delta.grad.data.zero_()
            # batch update
            # momentum
            if Momentum:
                batch_delta_grad = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0)
                if torch.norm(batch_delta_grad, p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9 * v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)


        #loss = cal_loss(loader_eval, model, delta.data, beta, loss_function)
        loss = cal_loss_targeted(loader_eval, model, delta.data, beta, loss_function, target_class)
        losses.append(torch.mean(loss.data).cpu())

        if (epoch + 1) % 10 == 0 or epoch == 1:
            torch.save(delta.data,
                       dir_uap + 'sga_' + '%d_%depoch_%dbatch.pth' % (img_num, epoch + 1, batch_size))

    return delta.data, losses



def uap_sga(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
            uap_init=None, batch_size=None, minibatch=10, loader_eval=None, dir_uap=None, center_crop=224, iter=4,
            Momentum=0, img_num=10000):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    minibatch   minibatch for SGA
    iter        iteration number (K) in the inner iteration
    center_crop image size
    Momentum    momentum item (default is false)

    log output
    batch_size  batch size
    loader_eval evaluation dataloader
    dir_uap     save patch
    img_num     total image num
    '''
    model.eval()
    DEVICE = torch.device("cuda:0")
    np.random.seed(0)
    if uap_init is None:
        batch_delta = torch.zeros(batch_size, 3, center_crop, center_crop)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    # loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

    batch_delta.requires_grad_()
    v = 0
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay
        for i, data in enumerate(loader):
            x_val = data
            with torch.no_grad():
                outputs_ori = model(x_val.cuda())
                _, target_label = torch.max(outputs_ori, 1)

            num = x_val.shape[0]
            k = iter
            noise_inner_all = torch.zeros(k * num // minibatch, 3, center_crop, center_crop)
            delta_inner = delta.data
            for j in range(k * num // minibatch):
                label = np.random.choice(num, minibatch, replace=False)
                if j > 0 or i > 0 or epoch > 0:
                    batch_delta.grad.data.zero_()
                batch_delta.data = delta_inner.unsqueeze(0).repeat([minibatch, 1, 1, 1])
                perturbed = torch.clamp((x_val[label] + batch_delta).cuda(), 0, 1)
                outputs = model(perturbed)
                # loss function value
                if loss_function:
                    loss = clamped_loss(outputs, target_label[label].cuda())
                else:
                    loss = -torch.mean(outputs.gather(1, (target_label[label].cuda()).unsqueeze(1)).squeeze(1))

                loss.backward()
                grad_inner = batch_delta.grad.data.mean(dim=0)
                delta_inner = delta_inner + grad_inner.sign() * eps_step
                delta_inner = torch.clamp(delta_inner, -eps, eps)
                noise_inner_all[j, :, :, :] = grad_inner
                batch_delta.grad.data.zero_()
            # batch update
            # momentum
            if Momentum:
                batch_delta_grad = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0)
                if torch.norm(batch_delta_grad, p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9 * v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)

        loss = cal_loss(loader_eval, model, delta.data, beta, loss_function)
        losses.append(torch.mean(loss.data).cpu())

        if (epoch + 1) % 10 == 0 or epoch == 1:
            torch.save(delta.data,
                       dir_uap + 'sga_' + '%d_%depoch_%dbatch.pth' % (img_num, epoch + 1, batch_size))

    return delta.data, losses


def uap_sga_targeted(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
                     target_class=0,
                     uap_init=None, batch_size=None, minibatch=10, loader_eval=None, dir_uap=None, center_crop=224,
                     iter=4, Momentum=0, img_num=10000):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    minibatch   minibatch for SGA
    iter        iteration number (K) in the inner iteration
    center_crop image size
    Momentum    momentum item (default is false)

    log output
    batch_size  batch size
    loader_eval evaluation dataloader
    dir_uap     save patch
    img_num     total image num
    '''
    model.eval()
    DEVICE = torch.device("cuda:0")
    np.random.seed(0)
    if uap_init is None:
        batch_delta = torch.zeros(batch_size, 3, center_crop, center_crop)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    losses = []
    # loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss

    batch_delta.requires_grad_()

    v = 0
    for epoch in range(nb_epoch):
        print('epoch %i/%i' % (epoch + 1, nb_epoch))

        # perturbation step size with decay
        eps_step = eps * step_decay
        for i, (x_val, y_val) in enumerate(loader):
            x_val = x_val.cuda()
            y_val = y_val.cuda()
            target_class_label = torch.ones(x_val.shape[0], dtype=torch.int64) * target_class
            with torch.no_grad():
                outputs_ori = model(x_val)
                _, target_label = torch.max(outputs_ori, 1)

            num = x_val.shape[0]
            k = iter
            noise_inner_all = torch.zeros(k * num // minibatch, 3, center_crop, center_crop)
            delta_inner = delta.data
            for j in range(k * num // minibatch):
                label = np.random.choice(num, minibatch, replace=False)
                if j > 0 or i > 0 or epoch > 0:
                    batch_delta.grad.data.zero_()
                batch_delta.data = delta_inner.unsqueeze(0).repeat([minibatch, 1, 1, 1])
                perturbed = torch.clamp((x_val[label] + batch_delta.cuda()).cuda(), 0, 1)
                outputs = model(perturbed)
                # loss function value
                if loss_function:
                    loss = -clamped_loss(outputs, target_class_label[label].cuda())
                else:
                    loss = torch.mean(outputs.gather(1, (target_class_label[label].cuda())))

                loss.backward()
                grad_inner = batch_delta.grad.data.mean(dim=0)
                delta_inner = delta_inner + grad_inner.sign() * eps_step
                delta_inner = torch.clamp(delta_inner, -eps, eps)
                noise_inner_all[j, :, :, :] = grad_inner
                batch_delta.grad.data.zero_()
            # batch update
            # momentum
            if Momentum:
                batch_delta_grad = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0)
                if torch.norm(batch_delta_grad, p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9 * v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = torch.mean(noise_inner_all.detach().clone(), dim=0, keepdim=True).squeeze(0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)

        loss = cal_loss_targeted(loader_eval, model, delta.data, beta, loss_function, target_class)
        losses.append(torch.mean(loss.data).cpu())

        if (epoch + 1) % 10 == 0:
            torch.save(delta.data,
                       dir_uap + 'sga_' + '%d_%depoch_%dbatch.pth' % (img_num, epoch + 1, batch_size))

    return delta.data, losses