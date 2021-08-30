import os
import jax
from jax import jit, numpy as jnp
from dmp.dmp import DMP

from matplotlib import pyplot as plt
from datetime import datetime
from pathlib import Path

import imageio

from functools import partial

class ProblemCircle(object):
    def __init__(self, **kwargs):

        self.R = 1
        self.lr = 1e6
        n_dmps = 2 # 2D circle
        n_bfs = 50

        y0 = jnp.array([0, 1])
        goal = jnp.array([1, 0])

        self.dmp = DMP(
            n_dmps,
            n_bfs,
            y0=y0,
            goal=goal,
        )

        logdir_name = "{}_{}".format(
            "circle",
            datetime.now().strftime("%Y.%m.%d_%H:%M:%S"),
        )
        self.log_path = Path("./logs").joinpath(logdir_name)
        print(self.log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)


    def solve(self, ):
        w = self.init_w()
        for it in range(50):
            (loss, y_rec), grad = self.loss_and_grad(w)
            print(loss)
            print(grad)
            w = self.grad_step(w, grad)
            self.plot(y_rec, it)

        self.gen_gif()


    def plot(self, y_rec, iteration):
        x = [y[0] for y in y_rec]
        z = [y[1] for y in y_rec]
        plt.plot(x, z)
        plt.savefig(self.log_path.joinpath("{}.png".format(iteration)))
        plt.clf()

    def gen_gif(self, ):
        images = []
        for filename in sorted(Path(".").glob("./{}/*.png".format(self.log_path)), key=os.path.getmtime):
            images.append(imageio.imread(filename))
        imageio.mimsave('./{}/final.gif'.format(self.log_path), images)


    def loss_and_grad(self, w):

        def loss_fn(w):
            timesteps = self.dmp.timesteps

            y_rec = []

            y = self.dmp.y0
            dy = jnp.zeros_like(y)
            self.dmp.reset_state()

            loss = 0
            for _ in range(timesteps):
                y, dy, _ = self.dmp.step(w, y, dy)
                loss += (jnp.sqrt(jnp.sum(y ** 2)) - self.R) ** 2
                y_rec.append(y)

            loss /= timesteps
            return loss, y_rec

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, y_rec), grad = grad_fn(w)
        return (loss, y_rec), grad


    def grad_step(self, w, grad):
        w -= self.lr * grad
        return w


    def init_w(self, ):
        key = jax.random.PRNGKey(0)
        w = jax.random.normal(key, [self.dmp.n_dmps, self.dmp.n_bfs], dtype=jnp.float32)
        return w
