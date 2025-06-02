import torch


def linear(x, y, r=0.015):
    # w = 1 - torch.linalg.norm(x - y + 10e-10) / r
    w = 1 - torch.prod(torch.abs(x - y), dim=-1) / r
    # w = torch.where(w < 0.0, 0, w)
    return w


def exponential(x, y, r=0.015):
    d = (x - y) ** 2
    d = d.sum() / r**2
    d = torch.exp(-d)
    return d


def piecewise(x, y, r=0.015, sigma=1.0):
    # Define each piece of the function
    q = torch.linalg.norm(x - y) / r

    def piece1(q):
        return sigma * ((5 / 2 - q) ** 4 - 5 * (3 / 2 - q) ** 4 + 10 * (1 - q) ** 4)

    def piece2(q):
        return sigma * ((5 / 2 - q) ** 4 - 5 * (3 / 2 - q) ** 4)

    def piece3(q):
        return sigma * ((5 / 2 - q) ** 4)

    def piece4(_):
        return 0.0

    # Conditional logic for piecewise function
    return torch.cond(
        q < 1 / 2,
        lambda q: piece1(q),
        lambda q: torch.cond(
            q < 3 / 2,
            lambda q: piece2(q),
            lambda q: torch.cond(
                q < 5 / 2, lambda q: piece3(q), lambda q: piece4(q), q
            ),
            q,
        ),
        q,
    )
