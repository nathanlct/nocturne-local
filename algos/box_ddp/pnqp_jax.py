import jax
import jax.numpy as jnp
import jax.scipy as jscipy
from . import util


def eclamp(x, lower, upper):
    # In-place!!
    if type(lower) == type(x):
        assert x.shape == lower.shape

    if type(upper) == type(x):
        assert x.shape == upper.shape

    # I = x < lower
    # x = x.at[I].set(lower[I]) if not isinstance(lower, float) else lower

    # I = x > upper
    # x = x.at[I].set(upper[I]) if not isinstance(upper, float) else upper

    return jnp.clip(x, lower, upper)
# @profile
def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    GAMMA = 0.1
    n_batch = 1
    n, _ = H.shape
    pnqp_I = 1e-11*jnp.eye(n) #.type_as(H).expand_as(H)

    def obj(x):
        return 0.5 * x.T @ H @ x  + q @ x

    if x_init is None:
        if n == 1:
            new_x_init = -(1./H.squeeze(2))*q
        else:
            H_lu = H.lu()
            new_x_init = -q.unsqueeze(2).lu_solve(*H_lu).squeeze(2) # Clamped in the x assignment.
    else:
        # TODO(eugenevinitsky) this is not the way young padawan
        new_x_init = jnp.array(x_init) # Don't over-write the original x_init.

    x = eclamp(new_x_init, lower, upper)

    # Active examples in the batch.
    J = jnp.ones(n_batch, dtype=bool)
    m = 1
    i = 0
    def cond_fun(val):
        return jnp.logical_and(val[0] < n_iter, val[1] > 0)
    def body_fn(val):
        i, m, x, H_, H_lu_, If = val
        i += 1
        g = H @ x + q

        # TODO: Could clean up the types here.
        Ic = (((x == lower) & (g > 0)) | ((x == upper) & (g < 0))).astype(jnp.float32)
        If = 1-Ic

        Hff_I = jnp.outer(If.astype(jnp.float32), If.astype(jnp.float32)).astype(If.dtype)
        not_Hff_I = 1-Hff_I
        # Hfc_I = jnp.outer(If.float(), Ic.float()).type_as(If)

        g_ = jnp.array(g)
        g_ = jnp.where(Ic.astype(bool), 0., g_)
        H_ = jnp.array(H)
        H_ = jnp.where(not_Hff_I.astype(bool), 0.0, H_)
        H_ += pnqp_I

        if n == 1:
            dx = -(1./H_.squeeze(2))*g_
        else:
            H_lu_ = jscipy.linalg.lu_factor(H_)
            dx = jscipy.linalg.lu_solve(H_lu_, -g_)

        J = jnp.linalg.norm(dx, 2) >= 1e-4
        print(jax.lax.stop_gradient(J.sum()))
        m = jax.lax.stop_gradient(J.sum()) # Number of active examples in the batch.

        alpha = jnp.ones(n_batch, x.dtype)
        decay = 0.1
        max_armijo = GAMMA
        count = 0

        def line_search_cond(val):
            alpha, count, max_armijo, x = val
            return jnp.logical_and(max_armijo <= GAMMA, count < 10)
        
        def line_search_fn(val):
            # Crude way of smaking sure too much time isn't being spent
            # doing the line search.
            # assert count < 10
            alpha, count, max_armijo, x = val
            maybe_x = eclamp(x + alpha * dx, lower, upper)
            armijos = (GAMMA+1e-6)*jnp.ones(n_batch).astype(x.dtype)
            line_objective = (obj(x)-obj(maybe_x))/jnp.dot(g, x-maybe_x)
            # take the line_objective if J is true
            armijos = jnp.where(J > 0, line_objective, armijos)
            I = armijos <= GAMMA
            # decay if I is true
            alpha = jnp.where(I > 0, alpha * decay, alpha)
            max_armijo = jnp.max(armijos)
            count += 1
            return [alpha, count, max_armijo, x]
        
        _, _, _, maybe_x = jax.lax.while_loop(line_search_cond, line_search_fn, [alpha, count, max_armijo, x])

        x = maybe_x
        return [i, m, x, H_, H_lu_, If]
    
    # run the loop until either n_iters is reached or m == 0
    i, m, x, H_, H_lu_, If = jax.lax.while_loop(cond_fun, body_fn, [i, m, x, H, jscipy.linalg.lu_factor(H), jnp.ones_like(x)])
    import ipdb; ipdb.set_trace()
    return x, H_ if n == 1 else H_lu_, If, i
