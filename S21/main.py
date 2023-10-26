import torch

from gpt import get_batch, decode, estimate_loss, GPTLanguageModel

def train_gpt(config):
    model = GPTLanguageModel()
    m = model.to(config.device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(m)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(m.state_dict(), config.save_path)
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
    open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

