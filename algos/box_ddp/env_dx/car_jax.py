from jax import jit, jacfwd, jacrev, hessian, lax
import jax.numpy as np
from jax.scipy.special import logsumexp
import jax

import logging
import imageio

from cfgs.config import PROJECT_PATH

np.set_printoptions(precision=3)

# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as onp

import hydra
import pickle
import matplotlib.pyplot as plt

plt.style.use("seaborn")
plt.rcParams['figure.figsize'] = [5, 5]

from tqdm.auto import tqdm

import time

import os

os.environ["DISPLAY"] = ":0.0"

from algos.box_ddp.env_dx.car import CarDx
from algos.box_ddp.pnqp_jax import pnqp

DT = 0.1  # [s] delta time step, = 1/FPS_in_server
N_X = 47
N_U = 2
TIME_STEPS = 5
dx = None


# #@jit
def discrete_dynamics(x, u):
    length = 20.0  # TODO(remove hardcoding)
    # accel = u[:, 0].unsqueeze(1)
    # turn = u[:, 1].unsqueeze(1)
    accel = u[0]
    turn = u[1]
    heading = x[dx.ego_heading_index] * jax.numpy.pi / 180
    speed = x[dx.ego_speed_index]
    slipAngle = jax.numpy.arctan(jax.numpy.tan(turn) / 2.0)
    dHeading = speed * jax.numpy.sin(turn) / length
    dX = speed * jax.numpy.cos(heading + slipAngle)
    dY = speed * jax.numpy.sin(heading + slipAngle)
    # heading update
    ego_heading_vec = jax.numpy.zeros_like(x)
    ego_heading_vec = ego_heading_vec.at[dx.ego_heading_index].set(1)
    x = x + ego_heading_vec * (dHeading * 180 / jax.numpy.pi) * dx.dt
    # speed updates
    ego_speed_vec = jax.numpy.zeros_like(x)
    ego_speed_vec = ego_speed_vec.at[dx.ego_speed_index].set(1)
    x = x + ego_speed_vec * accel * dx.dt
    # position update
    # TODO(eugenevinitsky) vectorize
    ego_pos_vec = jax.numpy.zeros_like(x)
    ego_pos_vec = ego_pos_vec.at[dx.ego_pos_index[0]].set(1)
    x = x + ego_pos_vec * dX * dx.dt
    ego_pos_vec = jax.numpy.zeros_like(x)
    ego_pos_vec = ego_pos_vec.at[dx.ego_pos_index[1]].set(1)
    x = x + ego_pos_vec * dY * dx.dt

    # now do the position updates for all the other vehicles
    # we assume that they are not going to change their trajectories
    # TODO(eugenevinitsky) vectorize
    if hasattr(dx, 'vehicle_pos_indices'):
        for i in range(len(dx.vehicle_heading_indices)):
            heading = x[dx.vehicle_heading_indices[i]] * jax.numpy.pi / 180
            speed = x[dx.vehicle_speed_indices[i]]
            dX = speed * jax.numpy.cos(heading + slipAngle)
            dY = speed * jax.numpy.sin(heading + slipAngle)
            veh_pos_unit_vector = jax.numpy.zeros_like(x)
            veh_pos_unit_vector = veh_pos_unit_vector.at[
                dx.vehicle_pos_indices[2 * i]].set(1)
            x = x + veh_pos_unit_vector * dX * dx.dt
            veh_pos_unit_vector = jax.numpy.zeros_like(x)
            veh_pos_unit_vector = veh_pos_unit_vector.at[
                dx.vehicle_pos_indices[2 * i + 1]].set(1)
            x = x + veh_pos_unit_vector * dY * dx.dt

    # now we recompute all the distances
    # goal dist
    new_goal_dist = jax.numpy.linalg.norm(x[dx.ego_pos_index] -
                                          x[dx.goal_pos_index])
    goal_dist_unit_vector = jax.numpy.zeros_like(x)
    goal_dist_unit_vector = goal_dist_unit_vector.at[dx.goal_dist_index].set(1)
    x = (1 - goal_dist_unit_vector) * x + goal_dist_unit_vector * new_goal_dist

    # dist to all the other vehicles
    if hasattr(dx, 'vehicle_pos_indices'):
        for i in range(len(dx.vehicle_heading_indices)):
            new_dist = jax.numpy.linalg.norm(
                x[dx.ego_pos_index] - x[dx.vehicle_pos_indices[2 * i:2 *
                                                               (i + 1)]])
            veh_dist_unit_vector = jax.numpy.zeros_like(x)
            veh_dist_unit_vector = veh_dist_unit_vector.at[
                dx.vehicle_dist_indices[i]].set(1)
            x = (1 -
                 veh_dist_unit_vector) * x + veh_dist_unit_vector * new_dist

    # now do this for all the object distances, here we need to recompute the closest point
    # frequently
    # TODO(eugenevinitsky) there is a bug here, objects get further as we turn towards them
    for i in range(len(dx.road_dist_indices)):
        # TODO(eugenevinitsky) remove the length hardcoding
        length = 360
        road_pos_center = x[dx.road_pos_indices[2 * i:2 * (i + 1)]]
        heading = x[dx.road_heading_indices[i]] * np.pi / 180
        endpoint_1 = jax.numpy.hstack(
            ((road_pos_center[0] + length * jax.numpy.cos(heading) / 2),
             (road_pos_center[1] + length * jax.numpy.sin(heading) / 2)))
        endpoint_2 = jax.numpy.hstack(
            ((road_pos_center[0] - length * jax.numpy.cos(heading) / 2),
             (road_pos_center[1] - length * jax.numpy.sin(heading) / 2)))
        ego_pos = x[dx.ego_pos_index]
        # find the closest point on the line and project
        proj_len = jax.numpy.dot((ego_pos - endpoint_1),
                                 (endpoint_2 - endpoint_1)) / (length**2)
        t = jax.numpy.maximum(
            jax.numpy.zeros_like(proj_len),
            jax.numpy.minimum(jax.numpy.ones_like(proj_len), proj_len))
        projection = endpoint_1 + t * (endpoint_2 - endpoint_1)
        new_dist = jax.numpy.linalg.norm(projection - ego_pos)

        road_dist_unit_vector = jax.numpy.zeros_like(x)
        road_dist_unit_vector = road_dist_unit_vector.at[
            dx.road_dist_indices[i]].set(1)
        # TODO(eugenevinitsky) fix to allow batch
        # new_dist = torch.abs(torch.dot(normal_vec, x[0, dx.ego_pos_index]) + b)
        x = (1 - road_dist_unit_vector) * x + road_dist_unit_vector * new_dist

    # print(x[dx.road_dist_indices])
    return x


#@jit
def rollout(x0, u_trj):
    # TODO(eugenevinitsky) put back
    x_final, x_trj = jax.lax.scan(rollout_looper, x0, u_trj)
    # x_trj = []
    # for i in range(u_trj.shape[0]):
    #     x0, x0 = rollout_looper(x0, u_trj[i])
    #     x_trj.append(x0)
    # x_trj = np.stack(x_trj)
    return np.vstack((x0, x_trj))


#@jit
def rollout_looper(x_i, u_i):
    x_ip1 = discrete_dynamics(x_i, u_i)
    return x_ip1, x_ip1


#@jit
def cost_1step(x, u):
    # TODO(eugenevinitsky) remove hardcoding
    desired_distance = 13
    dist_goal_vec = jax.numpy.zeros_like(x)
    dist_goal_vec = dist_goal_vec.at[0].set(1)
    Q = jax.numpy.diag(dist_goal_vec)
    cost = jax.numpy.matmul(x, jax.numpy.matmul(Q, x.T))**0.5
    # now form the quadratic barrier around collisions with objects
    mask_vec = jax.numpy.zeros_like(x)
    if dx.road_dist_indices is not None:
        mask_vec = mask_vec.at[dx.road_dist_indices].set(1)
    if hasattr(dx, 'veh_dist_indices') and dx.veh_dist_indices is not None:
        mask_vec = mask_vec.at[dx.veh_dist_indices].set(1)
    # TODO(eugenevinitsky) actually shape the log barrier
    # TODO(eugenevinitsky) compute the second order approximation of this so I can use
    # quadcosts instead
    filtered_vec = jax.numpy.matmul(jax.numpy.diag(mask_vec), x.T)
    constraint_vec = -filtered_vec + desired_distance
    constraint_cost = jax.numpy.matmul(jax.numpy.diag(mask_vec),
                                       jax.numpy.exp(2 * constraint_vec))
    # print(filtered_vec[dx.road_dist_indices], constraint_cost[dx.road_dist_indices], constraint_cost.sum())
    cost = cost + constraint_cost.sum()

    # form a control cost
    cost = cost + 0.001 * jax.numpy.matmul(u, u.T)
    # TODO(eugenevinitsky) remove hardcoding
    return cost


#@jit
def cost_final(x):  # x.shape:(5), u.shape(2)
    # TODO(eugenevinitsky) remove hardcoding
    desired_distance = 13
    dist_goal_vec = jax.numpy.zeros_like(x)
    dist_goal_vec = dist_goal_vec.at[0].set(1)
    Q = jax.numpy.diag(dist_goal_vec)
    cost = jax.numpy.matmul(x, jax.numpy.matmul(Q, x.T))**0.5
    # now form the quadratic barrier around collisions with objects
    mask_vec = jax.numpy.zeros_like(x)
    if dx.road_dist_indices is not None:
        mask_vec = mask_vec.at[dx.road_dist_indices].set(1)
    if hasattr(dx, 'veh_dist_indices') and dx.veh_dist_indices is not None:
        mask_vec = mask_vec.at[dx.veh_dist_indices].set(1)
    # TODO(eugenevinitsky) actually shape the log barrier
    # TODO(eugenevinitsky) compute the second order approximation of this so I can use
    # quadcosts instead
    filtered_vec = jax.numpy.matmul(jax.numpy.diag(mask_vec), x.T)
    constraint_vec = -filtered_vec + desired_distance
    constraint_cost = jax.numpy.matmul(jax.numpy.diag(mask_vec),
                                       jax.numpy.exp(2 * constraint_vec))
    # print(filtered_vec[dx.road_dist_indices], constraint_cost[dx.road_dist_indices], constraint_cost.sum())
    cost = cost + constraint_cost.sum()
    return cost


#@jit
def cost_trj(x_trj, u_trj):
    total = 0.
    total, x_trj, u_trj = jax.lax.fori_loop(0, TIME_STEPS - 1, cost_trj_looper,
                                            [total, x_trj, u_trj])
    total += cost_final(x_trj[-1])

    return total


#@jit
def cost_trj_looper(i, input_):
    total, x_trj, u_trj = input_
    total += cost_1step(x_trj[i], u_trj[i])

    return [total, x_trj, u_trj]


# #@jit
# def derivative_init():
#     jac_l = jit(jacfwd(cost_1step, argnums=[0,1]))
#     hes_l = jit(hessian(cost_1step, argnums=[0,1]))
#     jac_l_final = jit(jacfwd(cost_final))
#     hes_l_final = jit(hessian(cost_final))
#     jac_f = jit(jacfwd(discrete_dynamics, argnums=[0,1]))

#     return jac_l, hes_l, jac_l_final, hes_l_final, jac_f


#@jit
def derivative_stage(x, u):  # x.shape:(5), u.shape(3)
    jac_l = jacfwd(cost_1step, argnums=[0, 1])
    hes_l = hessian(cost_1step, argnums=[0, 1])
    jac_f = jacfwd(discrete_dynamics, argnums=[0, 1])
    l_x, l_u = jac_l(x, u)
    (l_xx, l_xu), (l_ux, l_uu) = hes_l(x, u)
    f_x, f_u = jac_f(x, u)

    return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u


#@jit
def derivative_final(x):
    jac_l_final = jacfwd(cost_final)
    hes_l_final = hessian(cost_final)
    l_final_x = jac_l_final(x)
    l_final_xx = hes_l_final(x)

    return l_final_x, l_final_xx


#@jit
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = l_x + f_x.T @ V_x
    Q_u = l_u + f_u.T @ V_x

    Q_xx = l_xx + f_x.T @ V_xx @ f_x
    Q_ux = l_ux + f_u.T @ V_xx @ f_x
    Q_uu = l_uu + f_u.T @ V_xx @ f_u

    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


#@jit
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = -Q_uu_inv @ Q_u
    K = -Q_uu_inv @ Q_ux

    return k, K


#@jit
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu @ k
    V_xx = Q_xx + 2 * K.T @ Q_ux + K.T @ Q_uu @ K

    return V_x, V_xx


#@jit
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T @ k - 0.5 * k.T @ Q_uu @ k


#@jit
def forward_pass(x_trj, u_trj, k_trj, K_trj):
    u_trj = np.arcsin(np.sin(u_trj))

    x_trj_new = np.empty_like(x_trj)
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[0], x_trj[0])
    u_trj_new = np.empty_like(u_trj)

    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = lax.fori_loop(
        0, TIME_STEPS - 1, forward_pass_looper,
        [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new])

    return x_trj_new, u_trj_new


#@jit
def forward_pass_looper(i, input_):
    x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new = input_

    u_next = u_trj[i] + k_trj[i] + K_trj[i] @ (x_trj_new[i] - x_trj[i])
    u_trj_new = jax.ops.index_update(u_trj_new, jax.ops.index[i], u_next)

    x_next = discrete_dynamics(x_trj_new[i], u_trj_new[i])
    x_trj_new = jax.ops.index_update(x_trj_new, jax.ops.index[i + 1], x_next)

    return [x_trj, u_trj, k_trj, K_trj, x_trj_new, u_trj_new]


#@jit
def backward_pass(x_trj, u_trj):
    k_trj = np.empty_like(u_trj)
    K_trj = np.empty((TIME_STEPS - 1, N_U, N_X))
    expected_cost_redu = 0.
    pnqp_iters = 20
    V_x, V_xx = derivative_final(x_trj[-1])

    for i in range(TIME_STEPS - 1):
        V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, n_qp_iter = backward_pass_looper(
            i, [
                V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu,
                pnqp_iters
            ])
        import ipdb
        ipdb.set_trace()

    import ipdb
    ipdb.set_trace()
    # V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, n_qp_iter = lax.fori_loop(
    #     0, TIME_STEPS-1, backward_pass_looper, [V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, pnqp_iters]
    # )

    return k_trj, K_trj, expected_cost_redu


#@jit
def backward_pass_looper(i, input_):
    V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, n_iter = input_
    n = TIME_STEPS - 2 - i

    l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivative_stage(x_trj[n], u_trj[n])
    Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u,
                                         V_x, V_xx)
    lb = dx.lower[n] - u_trj[n]
    ub = dx.higher[n] - u_trj[n]
    import ipdb
    ipdb.set_trace()
    k, Qt_uu_free_LU, If, n_qp_iter = pnqp(Q_uu,
                                           Q_u,
                                           lb,
                                           ub,
                                           x_init=k_trj[n],
                                           n_iter=n_iter)
    Q_ux_ = np.array(Q_ux)
    Q_ux_ = np.where((1 - If).tile((*Q_x.shape, 1)).T, np.zeros_like(Q_ux),
                     Q_ux)
    # Q_ux_[(1-If).unsqueeze(2).repeat(1,1,Q_ux.size(2)).bool()] = 0
    if N_U == 1:
        # Bad naming, Qt_uu_free_LU isn't the LU in this case.
        K = -((1. / Qt_uu_free_LU) * Q_ux_)
    else:
        K = jax.scipy.linalg.lu_solve(Qt_uu_free_LU, -Q_ux_)
        # K = -Q_ux_.lu_solve(*Qt_uu_free_LU)
    k_trj = jax.ops.index_update(k_trj, jax.ops.index[n], k)
    K_trj = jax.ops.index_update(K_trj, jax.ops.index[n], K)
    V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
    expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)

    return [
        V_x, V_xx, k_trj, K_trj, x_trj, u_trj, expected_cost_redu, n_qp_iter
    ]


#@jit
def run_ilqr_main(x0, u_trj):
    global jac_l, hes_l, jac_l_final, hes_l_final, jac_f

    max_iter = 300

    x_trj = rollout(x0, u_trj)
    cost_trace = jax.ops.index_update(np.zeros((max_iter + 1)),
                                      jax.ops.index[0], cost_trj(x_trj, u_trj))

    # x_trj, u_trj, cost_trace = lax.fori_loop(
    #     1, max_iter+1, run_ilqr_looper, [x_trj, u_trj, cost_trace]
    # )
    for i in range(max_iter):
        x_trj, u_trj, cost_trace = run_ilqr_looper(i,
                                                   [x_trj, u_trj, cost_trace])
        import ipdb
        ipdb.set_trace()

    return x_trj, u_trj, cost_trace


#@jit
def run_ilqr_looper(i, input_):
    import ipdb
    ipdb.set_trace()
    x_trj, u_trj, cost_trace = input_
    k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj)
    x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)

    total_cost = cost_trj(x_trj_new, u_trj_new)

    x_trj, u_trj, cost_trace = lax.cond(
        pred=(cost_trace[i - 1] > total_cost),
        true_operand=[
            i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new
        ],
        true_fun=run_ilqr_true_func,
        false_operand=[i, cost_trace, x_trj, u_trj],
        false_fun=run_ilqr_false_func,
    )

    # max_regu = 10000.0
    # min_regu = 0.01

    # regu += jax.nn.relu(min_regu - regu)
    # regu -= jax.nn.relu(regu - max_regu)

    return [x_trj, u_trj, cost_trace]


#@jit
def run_ilqr_true_func(input_):
    i, cost_trace, total_cost, x_trj, u_trj, x_trj_new, u_trj_new = input_

    cost_trace = jax.ops.index_update(cost_trace, jax.ops.index[i], total_cost)
    x_trj = x_trj_new
    u_trj = u_trj_new

    return [x_trj, u_trj, cost_trace]


#@jit
def run_ilqr_false_func(input_):
    i, cost_trace, x_trj, u_trj = input_

    cost_trace = jax.ops.index_update(cost_trace, jax.ops.index[i],
                                      cost_trace[i - 1])

    return [x_trj, u_trj, cost_trace]


@hydra.main(config_path='../../../cfgs/', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        '[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
        datefmt='%m-%d %H:%M:%S')

    TIME_STEPS = 10
    run_iter = 20
    total_reward = 0
    render = True
    imgs = []

    onp.random.seed(1)

    # carla init
    global dx
    dx = CarDx(cfg, TIME_STEPS)
    if 'visible_objects' in dx.dict_key_order:
        dx.vehicle_dist_indices = np.array(dx.vehicle_dist_indices)
        dx.vehicle_pos_indices = np.array(dx.vehicle_pos_indices)
        dx.vehicle_heading_indices = np.array(dx.vehicle_heading_indices)
        dx.vehicle_speed_indices = np.array(dx.vehicle_speed_indices)
    dx.road_dist_indices = np.array(dx.road_dist_indices)
    dx.road_pos_indices = np.array(dx.road_pos_indices)
    dx.road_heading_indices = np.array(dx.road_heading_indices)
    dx.goal_dist_index = np.array(dx.goal_dist_index)
    dx.ego_pos_index = np.array(dx.ego_pos_index)
    dx.goal_pos_index = np.array(dx.goal_pos_index)
    dx.ego_speed_index = np.array(dx.ego_speed_index)
    dx.ego_heading_index = np.array(dx.ego_heading_index)
    dx.lower = jax.numpy.array(dx.lower.detach().numpy()[:, 0, :])
    dx.higher = jax.numpy.array(dx.higher.detach().numpy()[:, 0, :])

    total_reward = 0
    for i in range(run_iter):
        state = dx.state.copy()
        command_start = time.perf_counter()
        # recreate controller using updated u_init (kind of wasteful right?)

        u_trj = onp.random.randn(TIME_STEPS - 1, N_U) * 1e-8
        u_trj = np.array(u_trj)
        nominal_states, nominal_actions, nominal_objs = run_ilqr_main(
            state, u_trj)
        # import ipdb; ipdb.set_trace()
        action = nominal_actions[0]  # take first planned action
        elapsed = time.perf_counter() - command_start
        # TODO(eugenevinitsky) remove hardcoding
        action = np.array(action)
        # print()
        # heads up, 0.4 turns you left, -0.4 turns you right
        s, r, _, _ = dx.env.step({8: {'accel': action[0], 'turn': action[1]}})
        total_reward += r[8]
        # print(s[8]['road_objects'][dx.road_dist_indices - 7], r, action)
        if r[8] == 0:
            import ipdb
            ipdb.set_trace()
            break
        logger.debug(
            "action taken: %.4f cost received: %.4f time taken: %.5fs", action,
            -r[8], elapsed)
        if render:
            img = dx.env.render()
            imgs.append(img)

    imageio.mimsave(PROJECT_PATH / 'algos/box_ddp/env_dx' + '/render.gif',
                    imgs,
                    duration=0.1)

    logger.info("Total reward %f", total_reward)

    # TODO:
    # * check 0 speed issue
    # * optimize speed cost setting
    # * retrain dynamical model, perhaps with automatically collected data and boosting/weighing
    # * check uncertainty model and iLQG
    # * check Guided Policy Search


if __name__ == '__main__':
    main()