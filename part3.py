"""Part 3: K-means, agglomerative clustering, and PCA from scratch."""
import numpy as np

from common import OUTPUT_DIR, load_data, write_csv, write_results


def kmeans(z, k):
    step = len(z) // k
    centroids = z[np.arange(k) * step].copy()
    assignments = np.full(len(z), -1, dtype=int)
    while True:
        squared = np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        updated = np.argmin(squared, axis=1)
        unchanged = np.array_equal(updated, assignments)
        assignments = updated
        for cluster in range(k):
            members = z[assignments == cluster]
            if len(members):
                centroids[cluster] = np.mean(members, axis=0)
        if unchanged:
            break
    wcss = float(np.sum((z - centroids[assignments]) ** 2))
    return assignments, wcss


def hierarchical(distance_matrix, linkage):
    clusters = [[index] for index in range(len(distance_matrix))]
    merges = []
    while len(clusters) > 1:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_distances = [
                    distance_matrix[left, right]
                    for left in clusters[i] for right in clusters[j]
                ]
                if linkage == "single":
                    height = min(pair_distances)
                elif linkage == "complete":
                    height = max(pair_distances)
                else:
                    height = float(np.mean(pair_distances))
                key = (height, min(clusters[i]), min(clusters[j]))
                if best is None or key < best[0]:
                    best = (key, i, j, height)
        _, i, j, height = best
        cluster_a, cluster_b = clusters[i], clusters[j]
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a
        merges.append((cluster_a.copy(), cluster_b.copy(), float(height)))
        combined = sorted(cluster_a + cluster_b)
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in (i, j)]
        clusters.append(combined)
        clusters.sort(key=lambda cluster: min(cluster))
    heights = [merge[2] for merge in merges]
    largest_gap = int(np.argmax(np.diff(heights)))
    merges_below_cut = largest_gap + 1
    return merges, 6 - merges_below_cut, merges_below_cut


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    _, raw = load_data()
    x = raw[:, :5]
    y = raw[:, 7].astype(int)
    value = raw[:, 8]
    mean = np.mean(x, axis=0)
    sd = np.std(x, axis=0, ddof=1)
    z = (x - mean) / sd
    result = {}

    wcss = {}
    assignments_by_k = {}
    for k in range(2, 7):
        labels, wcss[k] = kmeans(z, k)
        assignments_by_k[k] = labels
        result[f"wcss_k{k}"] = wcss[k]
    elbow_k = next(
        k for k in (3, 4, 5)
        if wcss[k] - wcss[k + 1] < 0.5 * (wcss[k - 1] - wcss[k])
    )
    result["elbow_k"] = elbow_k

    labels = assignments_by_k[3]
    cluster_order = sorted(
        range(3), key=lambda cluster: float(np.mean(x[labels == cluster, 2]))
    )
    segment_labels = np.empty(len(labels), dtype=int)
    segment_data = []
    for rank, cluster in enumerate(cluster_order):
        segment_labels[labels == cluster] = rank
        members = raw[labels == cluster]
        size = len(members)
        result[f"size_segment_{chr(65 + rank)}"] = size
        segment_data.append((
            chr(65 + rank), size,
            *[float(np.mean(members[:, i])) for i in range(5)],
            float(np.mean(members[:, 7])), float(np.mean(members[:, 8])),
        ))
    write_csv(
        OUTPUT_DIR / "part3_segments.csv",
        ("segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4",
         "mean_X5", "outcome_rate", "mean_value"),
        segment_data,
    )

    supplied = np.array([
        [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
        [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
        [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
        [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
        [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
        [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
    ])
    hierarchical_rows = []
    for linkage in ("single", "complete", "average"):
        merges, cut_count, merges_below = hierarchical(supplied, linkage)
        result[f"{linkage}_clusters_at_cut"] = cut_count
        for merge_no, (a, b, height) in enumerate(merges, start=1):
            members_a = "+".join(f"E{i + 1}" for i in a)
            members_b = "+".join(f"E{i + 1}" for i in b)
            hierarchical_rows.append((linkage, merge_no, members_a, members_b, height))
    write_csv(
        OUTPUT_DIR / "part3_hierarchical.csv",
        ("linkage", "merge_no", "cluster_a", "cluster_b", "height"),
        hierarchical_rows,
    )

    covariance = z.T @ z / (len(z) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for component in range(eigenvectors.shape[1]):
        pivot = np.argmax(np.abs(eigenvectors[:, component]))
        if eigenvectors[pivot, component] < 0:
            eigenvectors[:, component] *= -1
    total = float(np.sum(eigenvalues))
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    result["eigenvalue_sum"] = total
    for index in range(5):
        result[f"eigenvalue_{index + 1}"] = eigenvalues[index]
        result[f"explained_ratio_{index + 1}"] = ratios[index]
        result[f"cumulative_ratio_{index + 1}"] = cumulative[index]
    result["components_for_80pct"] = int(np.flatnonzero(cumulative >= 0.8)[0] + 1)
    for index, loading in enumerate(eigenvectors[:, 0], start=1):
        result[f"pc1_loading_X{index}"] = loading
    for index, loading in enumerate(eigenvectors[:, 1], start=1):
        result[f"pc2_loading_X{index}"] = loading
    projection = z @ eigenvectors[:, :2]
    for row in range(3):
        result[f"row{row + 1}_pc1"] = projection[row, 0]
        result[f"row{row + 1}_pc2"] = projection[row, 1]
    for rank, letter in enumerate(("A", "B", "C")):
        result[f"mean_pc1_segment_{letter}"] = float(np.mean(projection[segment_labels == rank, 0]))

    integer_keys = (
        "elbow_k", "size_segment_A", "size_segment_B", "size_segment_C",
        "single_clusters_at_cut", "complete_clusters_at_cut", "average_clusters_at_cut",
        "components_for_80pct",
    )
    write_results(OUTPUT_DIR / "part3_results.txt", result, integer_keys=integer_keys)


if __name__ == "__main__":
    main()
