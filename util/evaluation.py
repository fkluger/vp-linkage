import numpy as np
import scipy.optimize
import sklearn.metrics


def single_eval(true_vps, estimated_vps, K_inverse, missing_vp_penalty=90.):

    true_num_vps = true_vps.shape[0]
    true_vds = (K_inverse * np.matrix(true_vps).T).T
    for vi in range(true_vds.shape[0]):
        true_vds[vi,:] /= np.maximum(np.linalg.norm(true_vds[vi,:]), 1e-16)

    estm_num_vps = estimated_vps.shape[0]
    num_missing_vps = true_num_vps-estm_num_vps
    num_vp_penalty = np.maximum(num_missing_vps, 0)

    estm_vds = (K_inverse * np.matrix(estimated_vps).T).T
    for vi in range(estm_vds.shape[0]):
        estm_vds[vi,:] /= np.maximum(np.linalg.norm(estm_vds[vi,:]), 1e-16)

    cost_matrix = np.arccos(np.abs(np.array(true_vds * estm_vds.T))) * 180. / np.pi

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)

    errors = []
    for ri, ci in zip(row_ind, col_ind):
        errors += [cost_matrix[ri,ci]]
    if missing_vp_penalty > 0:
        errors += [missing_vp_penalty for _ in range(num_vp_penalty)]

    return errors, num_missing_vps


def calc_auc(error_array, cutoff=10.):

    error_array = error_array.squeeze()
    error_array = np.sort(error_array)
    num_values = error_array.shape[0]

    plot_points = np.zeros((num_values, 2))

    midfraction = 1.

    for i in range(num_values):
        fraction = (i + 1) * 1.0 / num_values
        value = error_array[i]
        plot_points[i, 1] = fraction
        plot_points[i, 0] = value
        if i > 0:
            lastvalue = error_array[i - 1]
            if lastvalue < cutoff < value:
                midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

    if plot_points[-1, 0] < cutoff:
        plot_points = np.vstack([plot_points, np.array([cutoff, 1])])
    else:
        plot_points = np.vstack([plot_points, np.array([cutoff, midfraction])])

    sorting = np.argsort(plot_points[:, 0])
    plot_points = plot_points[sorting, :]

    auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= cutoff, 0],
                              plot_points[plot_points[:, 0] <= cutoff, 1])
    auc = auc / cutoff

    return auc, plot_points
