import numpy
def _fiber_distance_internal_use(fiber_r,fiber_a,fiber_s, fiber_array, threshold=0, distance_method='Mean', fiber_landmarks=None,
                                 landmarks=None, sigmasq=None):
    """ Compute the total fiber distance from one fiber to an array of
    many fibers.
    This function does not handle equivalent fiber representations,
    for that use fiber_distance, above.
    """
    if landmarks is not None:
        print("ERROR: Please use distance method Landmarks to compute landmark distances")

    fiber_array_r = fiber_array[:,:,0]
    fiber_array_a = fiber_array[:,:,1]
    fiber_array_s = fiber_array[:, :, 2]

    # compute the distance from this fiber to the array of other fibers
    ddx = fiber_array_r - fiber_r
    ddy = fiber_array_a - fiber_a
    ddz = fiber_array_s - fiber_s

    dx = numpy.square(ddx)
    dy = numpy.square(ddy)
    dz = numpy.square(ddz)

    # sum dx dx dz at each point on the fiber and sqrt for threshold
    # distance = numpy.sqrt(dx + dy + dz)
    distance = dx + dy + dz

    # threshold if requested
    if threshold:
        # set values less than threshold to 0
        distance = distance - threshold * threshold
        idx = numpy.nonzero(distance < 0)
        distance[idx] = 0

    if distance_method == 'Mean':
        # sum along fiber
        distance = numpy.sum(numpy.sqrt(distance), 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.shape[1])
        #print(npts)
        distance = distance / npts
        # for consistency with other methods we need to square this value
        #distance = numpy.square(distance)
    elif distance_method == 'Hausdorff':
        # take max along fiber
        distance = numpy.max(distance, 1)
    elif distance_method == 'MeanSquared':
        # sum along fiber
        distance = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = len(fiber_r)
        distance = distance / npts
    elif distance_method == 'StrictSimilarity':
        # for use in laterality
        # this is the product of all similarity values along the fiber
        # not truly a distance but it's easiest to compute here in this function
        # where we have all distances along the fiber
        # print "distance range :", numpy.min(distance), numpy.max(distance)
        #distance = distance_to_similarity(distance, sigmasq)
        # print "similarity range :", numpy.min(distance), numpy.max(distance)
        distance = numpy.prod(distance, 1)
        # print "overall similarity range:", numpy.min(distance), numpy.max(distance)
    elif distance_method == 'Mean_shape':

        # sum along fiber
        distance_square = distance
        distance = numpy.sqrt(distance_square)

        d = numpy.sum(distance, 1)
        # Remove effect of number of points along fiber (mean)
        npts = float(fiber_array.points_per_fiber)
        d = numpy.divide(d, npts)
        # for consistency with other methods we need to square this value
        d = numpy.square(d)

        distance_endpoints = (distance[:, 0] + distance[:, npts - 1]) / 2

        for i in numpy.linspace(0, numpy.size(distance, 0) - 1, numpy.size(distance, 0)):
            for j in numpy.linspace(0, numpy.size(distance, 1) - 1, numpy.size(distance, 1)):
                if distance[i, j] == 0:
                    distance[i, j] = 1
        ddx = numpy.divide(ddx, distance)
        ddy = numpy.divide(ddy, distance)
        ddz = numpy.divide(ddz, distance)
        # print ddx*ddx+ddy*ddy+ddz*ddz
        npts = float(fiber_array.points_per_fiber)
        angles = numpy.zeros([(numpy.size(distance)) / npts, npts * (npts + 1) / 2])
        s = 0
        n = numpy.linspace(0, npts - 1, npts)
        for i in n:
            m = numpy.linspace(0, i, i + 1)
            for j in m:
                angles[:, s] = (ddx[:, i] - ddx[:, j]) * (ddx[:, i] - ddx[:, j]) + (ddy[:, i] - ddy[:, j]) * (
                            ddy[:, i] - ddy[:, j]) + (ddz[:, i] - ddz[:, j]) * (ddz[:, i] - ddz[:, j])
                s = s + 1
        angles = (numpy.sqrt(angles)) / 2
        angle = numpy.max(angles, 1)
        # print angle.max()

        distance = 0.5 * d + 0.4 * d / (0.5 + 0.5 * (1 - angle * angle)) + 0.1 * distance_endpoints

    else:
        print("<similarity.py> throwing Exception. Unknown input distance method (typo?):", distance_method)
        raise Exception("unknown distance method")

    return distance

def fiber_distance(fiber, fiber_array, threshold=0, distance_method='Mean', fiber_landmarks=None, landmarks=None,
                   sigmasq=6400, bilateral=False):
    """
    Find pairwise fiber distance from fiber to all fibers in fiber_array.
    The Mean and MeanSquared distances are the average distance per
    fiber point, to remove scaling effects (dependence on number of
    points chosen for fiber parameterization). The Hausdorff distance
    is the maximum distance between corresponding points.
    input fiber should be class Fiber. fibers should be class FiberArray
    """
    fiber_r=fiber[:,0]
    fiber_a=fiber[:,1]
    fiber_s = fiber[:, 2]

    # get fiber in reverse point order, equivalent representation
    fiber_r_quiv=fiber_r[::-1]
    fiber_a_quiv = fiber_a[::-1]
    fiber_s_quiv = fiber_s[::-1]

    # compute pairwise fiber distances along fibers
    distance_1 = _fiber_distance_internal_use(fiber_r,fiber_a,fiber_s, fiber_array, threshold, distance_method, fiber_landmarks,
                                              landmarks, sigmasq)
    distance_2 = _fiber_distance_internal_use(fiber_r_quiv,fiber_a_quiv,fiber_s_quiv, fiber_array, threshold, distance_method, fiber_landmarks,
                                              landmarks, sigmasq)

    # choose the lowest distance, corresponding to the optimal fiber
    # representation (either forward or reverse order)
    if distance_method == 'StrictSimilarity':
        # for use in laterality
        # this is the product of all similarity values along the fiber
        distance = numpy.maximum(distance_1, distance_2)
    else:
        distance = numpy.minimum(distance_1, distance_2)

    if bilateral:
        fiber_r_ref = -fiber_r
        fiber_reflect=numpy.stack((fiber_r_ref,fiber_a,fiber_s)).transpose([1,0])
        # call this function again with the reflected fiber. Do NOT reflect again (bilateral=False) to avoid infinite loop.
        distance_reflect = fiber_distance(fiber_reflect, fiber_array, threshold, distance_method, fiber_landmarks,
                                          landmarks, sigmasq, bilateral=False)
        # choose the best distance, corresponding to the optimal fiber
        # representation (either reflected or not)
        if distance_method == 'StrictSimilarity':
            # this is the product of all similarity values along the fiber
            distance = numpy.maximum(distance, distance_reflect)
        else:
            distance = numpy.minimum(distance, distance_reflect)

    return distance





def fiber_pair_distance(fiber1,fiber2):
    fiber_r1=fiber1[:,0]
    fiber_a1=fiber1[:,1]
    fiber_s1 = fiber1[:, 2]
    fiber_r2=fiber2[:,0]
    fiber_a2=fiber2[:,1]
    fiber_s2 = fiber2[:, 2]
    ddx = fiber_r1 - fiber_r2
    ddy = fiber_a1- fiber_a2
    ddz = fiber_s1 - fiber_s2
    dx = numpy.square(ddx)
    dy = numpy.square(ddy)
    dz = numpy.square(ddz)
    distance = dx + dy + dz
    distance = numpy.sum(numpy.sqrt(distance))
    npts = len(dx)
    distance = distance / npts
    return distance

def fiber_pair_similarity(fiber1,fiber2):
    distance1=fiber_pair_distance(fiber1,fiber2)
    fiber1_equiv=fiber1
    fiber1_equiv[:,0]=fiber1[::-1,0]
    fiber1_equiv[:, 1] = fiber1[::-1, 1]
    fiber1_equiv[:, 2] = fiber1[::-1, 2]
    distance2 = fiber_pair_distance(fiber1_equiv, fiber2)
    distance = numpy.minimum(distance1, distance2)

    # roi_onehot1 = numpy.zeros([64])
    # roi_label1 = numpy.unique(roi1)
    # if 0 in roi_label1:
    #     index_0 = roi_label1 != 0
    #     roi_label1 = roi_label1[index_0]
    #     print(roi_label1)
    # roi_onehot1[(roi_label1).astype(int)]=1
    #
    # roi_onehot2 = numpy.zeros([64])
    # roi_label2 = numpy.unique(roi2)
    # if 0 in roi_label2:
    #     index_0 = roi_label2 != 0
    #     roi_label2 = roi_label2[index_0]
    # roi_onehot2[(roi_label2).astype(int)]=1
    # intersection = roi_onehot1 * roi_onehot2
    # smooth=1e-10
    # dis_roi = 1-(2 * intersection.sum() + smooth) / ( roi_onehot1.sum() +  roi_onehot2.sum() + smooth)

    #similarity = numpy.exp(-distance / 2000)
    #print('distance',distance1,distance2,similarity,distance/2000)
    #print('distance', distance1, distance2,distance)
    #print(distance,dis_roi)
    #distance = dis_roi*distance
    return  distance #similarity
