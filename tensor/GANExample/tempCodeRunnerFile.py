    # optimmize
    optim_G.zero_grad()
    loss_G.backward()
    optim_G.step()

    if epoch % 100 == 0:
        print(loss_D.item(), loss_G.item())