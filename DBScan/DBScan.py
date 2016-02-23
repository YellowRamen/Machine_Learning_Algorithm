class DBSCANing (object):
    def __init__(self, D):
        # initialize needed information
        self.dataset = D
        # adding in what I need as I go
        self.visited = []
        self.noise = []
        self.cluster = []
        self.label = []


    def DBSCAN (self, eps, MinPts):
        # for every point in data
        for p in self.dataset:
            # if the point is visited, we continue
            if p in self.visited:
                continue
            # otherwise...
            else:
                # since now we have visited the point, we mark it
                self.visited.append(p)
                # we get the neighbor points
                NeighborPts = self.regionQuery(p, eps)
                # if the size of neighbor points is less than minimum
                if len(NeighborPts) < MinPts:
                    # we count it as noise
                    self.noise.append(p)
                else:
                    C = []
                    # or expand cluster... go to next clustering
                    self.expandCluster(p, NeighborPts, C, eps, MinPts)


    def expandCluster(self, p, NeighborPts, C, eps, MinPts):
        C.append(p)
        # for points in neighbor points
        for pt in NeighborPts:
            # if the point is not visited
            if pt not in self.visited:
            #if T is False:
                # now we have visited, so mark it!
                self.visited.append(pt)
                # new neighbor points defined
                npt = self.regionQuery(pt, eps)
                # if the size is smaller or equal to MinPts
                if len(npt) >= MinPts:
                    for pts in npt:
                        # check if the points in npt is not in NeighborPts already
                        if pts not in NeighborPts:
                            # so we can join the points (NeighborPts = NeighborPts joined with npt)
                            NeighborPts.append(pts)
            for group in self.cluster:
                if pt not in group:
                    C.append(pt)

        self.cluster.append(C)


    def regionQuery(self, p, eps):
        # will be populated by pts that are in the same cluster
        in_group = []
        # go through each data point
        for data in self.dataset:
            # calcuate the distance
            dist = np.sqrt((data[0]-p[0])**2 + (data[1]-p[1])**2)
            # check if the distance is within epislon range
            if dist < eps:
                # populate into cluster
                in_group.append(data)
        return in_group


    def labels(self):
        # go through the points in dataset
        for pt in self.dataset:
            # go through each cluster
            for clusters in self.cluster:
                for i, item in enumerate(clusters):
                    # if the point belongs to a cluter
                    if pt in item:
                        # we tag it by the cluster number (0, 1, ... n)
                        self.label.append(i)
                    # if not in cluster it is probably in noise
                    else:
                        # if in noise
                        for item in self.noise:
                            if pt in item:
                                # we assign -1 as value
                                self.label.append(-1)
