from tqdm import trange
import time
from .functions import train_error, update_model, loss_and_grad


def train_loop(model_fn, params, train_data, test_data, optimizer, opt_state, args, log_file):
    for epoch in trange(args.epochs):

        if epoch == 1:
            # exclude compile time
            start = time.time()

        # Train model
        loss, gradient = loss_and_grad(model_fn, params, train_data[0], train_data[1])
        params, state = update_model(optimizer, gradient, params, opt_state)

        # Log loss
        if epoch % args.log_iter == 0:
            error = train_error(model_fn, params, test_data[0], test_data[1])
            log_file.write(f'Epoch: {epoch}, train loss: {loss}, test error: {error}\n')
            print(f'Epoch: {epoch}/{args.epochs} --> train loss: {loss:.8f}, test error: {error:.8f}')

    print("Training done")
    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime / (args.epochs - 1) * 1000):.2f}ms/iter.)')

    return params

