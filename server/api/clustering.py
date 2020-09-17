
def clusterize(crops, scores, n_results, dist):
    n_crops = len(crops)
    centers = [((crop[3] + crop[1]) / 2, (crop[2] + crop[0]) / 2) for crop in crops]
    
    dists = []
    for i_center in range(n_crops):
        for j_center in range(i_center, n_crops):
            dists.append((dist(centers[i_center], centers[j_center]), (i_center, j_center)))

    parents = list(range(n_crops))
    cur_classes = n_crops
    
    def get_head(c):
        if parents[c] != c:
            parents[c] = get_head(parents[c])
        return parents[c]

    for _, (i, j) in sorted(dists, key=lambda (x, _): x):
        if cur_classes == n_results:
            break
        i_head = get_head(i)
        j_head = get_head(j)
        if i_head != j_head:
            cur_classes -= 1
            parents[j_head] = i_head
    
    classes = {}
    for i in range(n_crops):
        head = get_head(i)
        if head not in classes:
            classes[head] = []
        
        classes[head].append(i)

    return [max(members, key=lambda k: scores[k]) for members in classes.values()]
