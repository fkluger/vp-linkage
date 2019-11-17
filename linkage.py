import numpy as np
from datasets import yud


def get_line_segments_from_indices(line_segments, indices):
    sampled_lines = line_segments[indices,:]
    return sampled_lines


def vanishing_point_from_lines(lines, weights=None):
    num_lines = lines.shape[0]

    if num_lines == 2 and weights is None:
        line1 = lines[0]
        line2 = lines[1]
        return np.cross(line1, line2)

    else:
        A = lines

        if weights is not None:
            W = np.diag(weights / np.max(weights))
            Mat = np.dot(W, A)
        else:
            Mat = A

        U, S, V = np.linalg.svd(Mat)

        V = V.T
        vp = np.squeeze(V[:, 2])
        vp /= np.linalg.norm(vp, ord=2)
        vp *= np.sign(vp[2])
        return vp


def sample_lines(line_segments, hom_lines, centroids, num_samples=2):
    num_lines = line_segments.shape[0]

    sampled_indices = np.random.choice(num_lines, num_samples, False)

    return line_segments[sampled_indices], hom_lines[sampled_indices], centroids[sampled_indices]


def consistency_measure(vp, centroid, endpoint):
    if endpoint.shape[0] == 2:
        endpoint = np.array([endpoint[0], endpoint[1], 1])

    if centroid.shape[0] == 2:
        centroid = np.array([centroid[0], centroid[1], 1])

    constrained_line = np.cross(centroid, vp)
    constrained_line /= np.linalg.norm(constrained_line[0:2])
    distance = np.abs(np.dot(endpoint, constrained_line).squeeze())
    return distance


def consistency_measure_angle(vp, centroids, lines):

    constrained_lines = np.cross(centroids, vp)
    constrained_lines /= np.expand_dims(np.linalg.norm(constrained_lines[:,0:2], axis=-1), -1)
    lines /= np.expand_dims(np.linalg.norm(lines[:,0:2], axis=-1), -1)
    distances = 1-np.abs(np.sum(lines[:,0:2] * constrained_lines[:,0:2], axis=-1))
    return distances


def build_preference_matrix_jlinkage(line_segments, lines, centroids, num_hypotheses, consensus_threshold=2.):

    num_lines = line_segments.shape[0]

    preference_mat = np.zeros((num_lines, num_hypotheses)).astype(np.int)

    for i_hypo in range(num_hypotheses):
        segment_samples, line_samples, centroid_samples = sample_lines(line_segments, lines, centroids, 2)
        vp_hypothesis = vanishing_point_from_lines(line_samples)
        distances = consistency_measure_angle(vp_hypothesis, centroids,
                                             lines)
        for i_line in range(num_lines):
            distance = distances[i_line]
            preference_mat[i_line, i_hypo] = 1 if distance <= consensus_threshold else 0

    return preference_mat


def build_preference_matrix_tlinkage(line_segments, lines, centroids, num_hypotheses, consensus_threshold=2.):

    num_lines = line_segments.shape[0]

    preference_mat = np.zeros((num_lines, num_hypotheses)).astype(np.float)

    for i_hypo in range(num_hypotheses):
        segment_samples, line_samples, centroid_samples = sample_lines(line_segments, lines, centroids, 2)
        vp_hypothesis = vanishing_point_from_lines(line_samples)
        distances = consistency_measure_angle(vp_hypothesis, centroids,
                                              lines)
        for i_line in range(num_lines):
            distance = distances[i_line]
            preference_mat[i_line, i_hypo] = np.exp(-distance/consensus_threshold) if distance < 5*consensus_threshold else 0

    return preference_mat


def jaccard_distance(set_a, set_b):
    intersection = np.sum(np.logical_and(set_a, set_b))
    union = np.sum(np.logical_or(set_a, set_b))
    return 1.*intersection/np.maximum(union, 1e-8)


def jaccard_distance_binary(set_a, set_b):
    intersection = np.logical_and(set_a, set_b)
    sum_intersection = np.sum(intersection)

    return 1 if sum_intersection > 0 else 0


def jlinkage_clustering(preference_mat):

    keep_clustering = True
    cluster_step = 0

    num_clusters = preference_mat.shape[0]
    clusters = [[i] for i in range(num_clusters)]

    while keep_clustering:
        smallest_distance = 0
        best_combo = None
        keep_clustering = False

        num_clusters = preference_mat.shape[0]

        for i in range(num_clusters):
            for j in range(i):
                set_a = preference_mat[i]
                set_b = preference_mat[j]
                intersection = np.count_nonzero(np.logical_and(set_a, set_b))
                union = np.count_nonzero(np.logical_or(set_a, set_b))
                distance = 1.*intersection/np.maximum(union, 1e-8)

                if distance > smallest_distance:
                    keep_clustering = True
                    smallest_distance = distance
                    best_combo = (i,j)

        if keep_clustering:
            clusters[best_combo[0]] += clusters[best_combo[1]]
            clusters.pop(best_combo[1])
            set_a = preference_mat[best_combo[0]]
            set_b = preference_mat[best_combo[1]]
            merged_set = np.logical_and(set_a, set_b)
            preference_mat[best_combo[0]] = merged_set
            preference_mat = np.delete(preference_mat, best_combo[1], axis=0)
            cluster_step += 1

    print("clustering finished after %d steps" % cluster_step)

    return preference_mat, clusters


def tlinkage_clustering(preference_mat):

    keep_clustering = True
    cluster_step = 0

    num_clusters = preference_mat.shape[0]
    clusters = [[i] for i in range(num_clusters)]

    while keep_clustering:
        smallest_distance = 1.
        best_combo = None
        keep_clustering = False

        num_clusters = preference_mat.shape[0]

        preference_mat = np.matrix(preference_mat)
        preference_mat_selfprod = preference_mat * preference_mat.T

        for i in range(num_clusters):
            for j in range(i):
                distance = 1 - preference_mat_selfprod[i,j] / (preference_mat_selfprod[i,i] + preference_mat_selfprod[j,j] - preference_mat_selfprod[i,j])

                if distance < smallest_distance:
                    keep_clustering = True
                    smallest_distance = distance
                    best_combo = (i,j)

        if keep_clustering:
            clusters[best_combo[0]] += clusters[best_combo[1]]
            clusters.pop(best_combo[1])
            set_a = preference_mat[best_combo[0]]
            set_b = preference_mat[best_combo[1]]
            merged_set = np.minimum(set_a, set_b)
            preference_mat[best_combo[0]] = merged_set
            preference_mat = np.delete(preference_mat, best_combo[1], axis=0)
            cluster_step += 1

    print("clustering finished after %d steps" % cluster_step)

    return preference_mat, clusters


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import util.evaluation

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hyps', type=int, default=100, help='number of hypotheses')
    parser.add_argument('--threshold', '-t', type=float, default=0.001, help='inlier threshold')
    parser.add_argument('--runs', type=int, default=1, help='number of runs')
    parser.add_argument('--dataset', type=str, default="yud", help='dataset to use')
    parser.add_argument('--dataset_path', type=str, default="/tnt/data/scene_understanding/YUD", help='path to dataset')
    parser.add_argument('--tlinkage', dest='tlinkage', action='store_true',
                        help='use T-Linkage instead of J-Linkage',
                        default=False)
    parser.add_argument('--plot_recall', dest='plot_recall', action='store_true',
                        help='Plot recall curves of all runs', default=False)
    parser.add_argument('--plot_results', dest='plot_results', action='store_true',
                        help='Visualise results for every image', default=False)
    opt = parser.parse_args()

    if opt.dataset == 'yud':
        dataset = yud.YUDVP(opt.dataset_path, split='test', return_images=True)
    else:
        assert False, "unknown dataset"

    colourmap = plt.get_cmap('jet')

    results = []
    all_aucs = []

    for si in range(opt.runs):

        all_errors = []
        all_missing_vps = []

        for idx in range(len(dataset)):

            print("image ", idx)

            data = dataset[idx]['line_segments']

            p1 = data[:, 0:3]
            p2 = data[:, 3:6]
            segments = np.hstack([p1, p2])
            lines = data[:, 6:9]
            centroids = data[:, 9:12]

            if opt.tlinkage:
                initital_preference_mat = build_preference_matrix_tlinkage(segments, lines, centroids, opt.hyps,
                                                                           consensus_threshold=opt.threshold)
                clustered_preference_mat, clusters = tlinkage_clustering(initital_preference_mat.copy())
            else:
                initital_preference_mat = build_preference_matrix_jlinkage(segments, lines, centroids, opt.hyps,
                                                                           consensus_threshold=opt.threshold)
                clustered_preference_mat, clusters = jlinkage_clustering(initital_preference_mat.copy())

            print("# detected VPs/clusters: ", len(clusters))

            estimated_vps = []

            all_distances = []
            all_consensus = []

            for ci, cluster in enumerate(clusters):
                sampled_lines = get_line_segments_from_indices(lines, cluster)
                vp = vanishing_point_from_lines(sampled_lines)
                distances = consistency_measure_angle(vp, centroids, lines)
                consensus = np.zeros_like(distances)
                if opt.tlinkage:
                    for li in range(distances.shape[0]):
                        consensus[li] = np.exp(-distances[li] / opt.threshold) if \
                            distances[li] < 5 * opt.threshold else 0
                else:
                    for li in range(distances.shape[0]):
                        consensus[li] = 1 if distances[li] <= opt.threshold else 0

                all_distances += [distances]
                all_consensus += [consensus]
                vp /= vp[2]
                estimated_vps += [vp]

            all_consensus = np.vstack(all_consensus)
            sorting = []
            for mi in range(all_consensus.shape[0]):
                inlier_counts = np.sum(all_consensus, axis=1)
                best = np.argmax(inlier_counts)
                sorting += [best]

                for i in range(all_consensus.shape[0]):
                    all_consensus[i] = np.maximum(all_consensus[i], all_consensus[best])
                all_consensus[best] = 0

            sorting = np.array(sorting)

            if opt.plot_results:
                plt.figure()
                for ci, cluster in enumerate(clusters):
                    colour = colourmap((ci+1)*1./len(clusters))
                    sampled_lines = get_line_segments_from_indices(lines, cluster)
                    for i in cluster:
                        plt.plot([segments[i, 0], segments[i, 3]],
                                 [-segments[i, 1], -segments[i, 4]], '-', c=colour)
                plt.show()

            estimated_vps = np.vstack(estimated_vps)
            estimated_vps = estimated_vps[sorting]
            true_vps = dataset[idx]['VPs']
            errors, missing_vps = util.evaluation.single_eval(dataset.K_inv, true_vps,
                                                              estimated_vps[0:true_vps.shape[0]])
            all_missing_vps += [missing_vps]
            print("errors: ", errors)
            all_errors += errors

        auc, plot_points = util.evaluation.calc_auc(np.array(all_errors), cutoff=10.)
        auc *= 100.

        print("AUC: ", auc)

        if len(all_missing_vps) > 0:
            all_missing_vps = np.array(all_missing_vps)
            all_missing_vps[all_missing_vps > 0] = 0
            print("to many vps: ", np.mean(all_missing_vps))
            missing_vps = np.mean(all_missing_vps)
        else:
            missing_vps = 0

        results += [{'auc': auc, 'missing_vps': missing_vps, 'plot_points': plot_points}]

        all_aucs += [auc]

    print("AUCs: mean / std. / median: ", np.mean(all_aucs), np.std(all_aucs), np.median(all_aucs))

    if opt.plot_recall:
        plt.figure()
        plt.suptitle("dataset: %s" % opt.dataset)
        for ri, result in enumerate(results):
            plot_points = result['plot_points']
            auc = result['auc']
            label = "AUC: %.2f" % auc

            ax = plt.subplot2grid((1, len(results)), (0, ri))

            ax.plot(plot_points[:, 0], plot_points[:, 1], 'b-', label=label)

            ax.set_xlim([0, 10])
            ax.set_ylim([0, 1])
            ax.legend()

        plt.show()


