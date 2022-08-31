""" fibers.py

This module contains code for representation of tractography using a
fixed-length parameterization.

class FiberArray

"""

import numpy
import vtk
import time
import  whitematteranalysis as wma


class Fiber:
    """A class for fiber tractography data, represented with a fixed length"""

    def __init__(self):
        self.r = None
        self.a = None
        self.s = None
        self.points_per_fiber = None
        self.hemisphere_percent_threshold = 0.95
        
    def get_equivalent_fiber(self):
        """ Get the reverse order of current line (trajectory), as the
        fiber can be equivalently represented in either order."""
        
        fiber = Fiber()

        fiber.r = self.r[::-1]
        fiber.a = self.a[::-1]
        fiber.s = self.s[::-1]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_reflected_fiber(self):
        """ Returns reflected version of current fiber by reflecting
        fiber across midsagittal plane. Just sets output R coordinate to -R."""
 
        fiber = Fiber()

        fiber.r = - self.r
        fiber.a = self.a
        fiber.s = self.s

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def match_order(self, other):
        """ Reverse order of fiber to match this one if needed """
        # compute correlation
        corr = numpy.multiply(self.r, other.r) + \
            numpy.multiply(self.a, other.a) + \
            numpy.multiply(self.s, other.s)

        other2 = other.get_equivalent_fiber()
        corr2 = numpy.multiply(self.r, other2.r) + \
            numpy.multiply(self.a, other2.a) + \
            numpy.multiply(self.s, other2.s)
        
        if numpy.sum(corr) > numpy.sum(corr2):
            return other
        else:
            return other2
        
    def __add__(self, other):
        """This is the + operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r + other_matched.r
        fiber.a = self.a + other_matched.a
        fiber.s = self.s + other_matched.s
        return fiber
    
    def __div__(self, other):
        """ This is to divide a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.divide(self.r, other)
        fiber.a = numpy.divide(self.a, other)
        fiber.s = numpy.divide(self.s, other)
        return fiber

    def __mul__(self, other):
        """ This is to multiply a fiber by a number"""
        fiber = Fiber()
        fiber.r = numpy.multiply(self.r, other)
        fiber.a = numpy.multiply(self.a, other)
        fiber.s = numpy.multiply(self.s, other)
        return fiber
    
    def __subtract__(self, other):
        """This is the - operator for fibers"""
        other_matched = self.match_order(other)
        fiber = Fiber()
        fiber.r = self.r - other_matched.r
        fiber.a = self.a - other_matched.a
        fiber.s = self.s - other_matched.s
        #fiber.r = self.r + other_matched.r
        #fiber.a = self.a + other_matched.a
        #fiber.s = self.s + other_matched.s
        return fiber
    
class FiberArray:

    """A class for arrays of fiber tractography data, represented with
    a fixed length"""

    def __init__(self):
        # parameters
        self.points_per_fiber = 10
        self.verbose = 0

        # fiber data
        self.fiber_array_r = None
        self.fiber_array_a = None
        self.fiber_array_s = None

        # output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
        self.fiber_hemisphere = None
        self.hemispheres = False
        
        # output boolean arrays for each hemisphere and callosal fibers
        self.is_left_hem = None
        self.is_right_hem = None
        self.is_commissure = None

        # output indices of each type above
        self.index_left_hem = None
        self.index_right_hem = None
        self.index_commissure = None
        self.index_hem = None

        # output totals of each type also
        self.number_of_fibers = 0
        self.number_left_hem = None
        self.number_right_hem = None
        self.number_commissure = None

    def __str__(self):
        output = "\n points_per_fiber\t" + str(self.points_per_fiber) \
            + "\n number_of_fibers\t\t" + str(self.number_of_fibers) \
            + "\n fiber_hemisphere\t\t" + str(self.fiber_hemisphere) \
            + "\n verbose\t" + str(self.verbose)

        return output

    def _calculate_line_indices(self, input_line_length, output_line_length):
        """ Figure out indices for downsampling of polyline data.

        The indices include the first and last points on the line,
        plus evenly spaced points along the line.  This code figures
        out which indices we actually want from a line based on its
        length (in number of points) and the desired length.

        """

        # this is the increment between output points
        step = (input_line_length - 1.0) / (output_line_length - 1.0)

        # these are the output point indices (0-based)
        ptlist = []
        for ptidx in range(0, output_line_length):
        #print(ptidx*step)
            ptlist.append(ptidx * step)

        # test
        if __debug__:
            # this tests we output the last point on the line
            #test = ((output_line_length - 1) * step == input_line_length - 1)
            test = (round(ptidx*step) == input_line_length-1)
            if not test:
                print("<fibers.py> ERROR: fiber numbers don't add up.")
                print(step)
                print(input_line_length)
                print(output_line_length)
                print(test)
                raise AssertionError

        return ptlist

    def get_fiber(self, fiber_index):
        """ Return fiber number fiber_index. Return value is class
        Fiber."""

        fiber = Fiber()
        fiber.r = self.fiber_array_r[fiber_index, :]
        fiber.a = self.fiber_array_a[fiber_index, :]
        fiber.s = self.fiber_array_s[fiber_index, :]

        fiber.points_per_fiber = self.points_per_fiber

        return fiber

    def get_equivalent_fiber(self, fiber_index):
        """ Return equivalent version of fiber number
        fiber_index. Return value is class Fiber. Gets the reverse
        order of line (trajectory), as the fiber can be equivalently
        represented in either order."""
  
        fiber = self.get_fiber(fiber_index)

        return fiber.get_equivalent_fiber()

    def get_fibers(self, fiber_indices):
        """ Return FiberArray containing subset of data corresponding
        to fiber_indices"""
        
        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        if self.fiber_hemisphere is not None:
            # Output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
            fibers.fiber_hemisphere = self.fiber_hemisphere[fiber_indices]

            # output boolean arrays for each hemisphere and callosal fibers
            fibers.is_left_hem = self.is_left_hem[fiber_indices]
            fibers.is_right_hem = self.is_right_hem[fiber_indices]
            fibers.is_commissure = self.is_commissure[fiber_indices]

            # calculate indices of each type above
            fibers.index_left_hem = numpy.nonzero(fibers.is_left_hem)[0]
            fibers.index_right_hem = numpy.nonzero(fibers.is_right_hem)[0]
            fibers.index_commissure = numpy.nonzero(fibers.is_commissure)[0]
            fibers.index_hem = \
                numpy.nonzero(fibers.is_left_hem | fibers.is_right_hem)[0]

            # output totals of each type also
            fibers.number_left_hem = len(fibers.index_left_hem)
            fibers.number_right_hem = len(fibers.index_right_hem)
            fibers.number_commissure = len(fibers.index_commissure)

            # test
            if __debug__:
                test = fibers.number_of_fibers == \
                    (fibers.number_left_hem + fibers.number_right_hem \
                         + fibers.number_commissure)
                if not test:
                    print("<fibers.py> ERROR: fiber numbers don't add up.")
                    raise AssertionError

        return fibers

    def get_oriented_fibers(self, fiber_indices, order):
        """Return FiberArray containing subset of data corresponding to
        fiber_indices. Order fibers according to the array (where 0 is no

        change, and 1 means to reverse the order and return the
        equivalent fiber)
        """

        fibers = FiberArray()

        fibers.number_of_fibers = len(fiber_indices)

        # parameters
        fibers.points_per_fiber = self.points_per_fiber
        fibers.verbose = self.verbose

        # fiber data
        fibers.fiber_array_r = self.fiber_array_r[fiber_indices]
        fibers.fiber_array_a = self.fiber_array_a[fiber_indices]
        fibers.fiber_array_s = self.fiber_array_s[fiber_indices]

        # swap orientation as requested
        for (ord, fidx) in zip(order, range(fibers.number_of_fibers)):
            if ord == 1:
                f2 = fibers.get_equivalent_fiber(fidx)
                # replace it in the array
                fibers.fiber_array_r[fidx,:] = f2.r
                fibers.fiber_array_a[fidx,:] = f2.a
                fibers.fiber_array_s[fidx,:] = f2.s

        if self.fiber_hemisphere is not None:
            # Output arrays indicating hemisphere/callosal (L,C,R= -1, 0, 1)
            fibers.fiber_hemisphere = self.fiber_hemisphere[fiber_indices]

            # output boolean arrays for each hemisphere and callosal fibers
            fibers.is_left_hem = self.is_left_hem[fiber_indices]
            fibers.is_right_hem = self.is_right_hem[fiber_indices]
            fibers.is_commissure = self.is_commissure[fiber_indices]

            # calculate indices of each type above
            fibers.index_left_hem = numpy.nonzero(fibers.is_left_hem)[0]
            fibers.index_right_hem = numpy.nonzero(fibers.is_right_hem)[0]
            fibers.index_commissure = numpy.nonzero(fibers.is_commissure)[0]
            fibers.index_hem = \
                numpy.nonzero(fibers.is_left_hem | fibers.is_right_hem)[0]

            # output totals of each type also
            fibers.number_left_hem = len(fibers.index_left_hem)
            fibers.number_right_hem = len(fibers.index_right_hem)
            fibers.number_commissure = len(fibers.index_commissure)

            # test
            if __debug__:
                test = fibers.number_of_fibers == \
                    (fibers.number_left_hem + fibers.number_right_hem \
                         + fibers.number_commissure)
                if not test:
                    print("<fibers.py> ERROR: fiber numbers don't add up.")
                    raise AssertionError

        return fibers

    def convert_from_polydata(self, input_vtk_polydata, points_per_fiber=None,gifti=None,cifti=None,dir_flag=False,data='HCP'):

        """Convert input vtkPolyData to the fixed length fiber
        representation of this class.

        The polydata should contain the output of tractography.

        The output is downsampled fibers in array format and
        hemisphere info is also calculated.

        """

        # points used in discretization of each trajectory
        if points_per_fiber is not None:
            self.points_per_fiber = points_per_fiber

        # line count. Assume all input lines are from tractography.
        self.number_of_fibers = input_vtk_polydata.GetNumberOfLines()

        if self.verbose:
            print("<fibers.py> Converting polydata to array representation. Lines:", \
                self.number_of_fibers)

        # allocate array number of lines by line length
        self.fiber_array_r = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_array_a = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_array_s = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_array_p = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_dir_r = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_dir_a = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_dir_s = numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))
        self.fiber_length = numpy.zeros(self.number_of_fibers)
        self.fiber_surface_ve = numpy.zeros((self.number_of_fibers,2))
        self.fiber_surface_dk = numpy.zeros((self.number_of_fibers, 2))
        self.fiber_surface_des = numpy.zeros((self.number_of_fibers, 2))
        self.fiber_roi= numpy.zeros((self.number_of_fibers,
                                          self.points_per_fiber))

        inpointdata = input_vtk_polydata.GetPointData()
        point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))
        roi_list = []
        roi_list1 = []
        roi_list2 = []
        roi_list3 = []
        roi_list4 = []
        roi_list5 = []
        roi_list6 = []
        for idx in point_data_array_indices:
            array = inpointdata.GetArray(idx)
            if array.GetName() == 'ROI_label':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list.append(roi_line)
                self.roi_list = roi_list
            elif array.GetName() == 'ROI_nei1':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list1.append(roi_line)
                self.roi_list1 = roi_list1
            elif array.GetName() == 'ROI_nei2':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list2.append(roi_line)
                self.roi_list2 = roi_list2
            elif array.GetName() == 'ROI_nei3':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list3.append(roi_line)
                self.roi_list3 = roi_list3
            elif array.GetName() == 'ROI_nei4':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    # print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list4.append(roi_line)
                self.roi_list4 = roi_list4
            elif array.GetName() == 'ROI_nei5':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    # print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list5.append(roi_line)
                self.roi_list5 = roi_list5
            elif array.GetName() == 'ROI_nei6':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    # print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    roi_list6.append(roi_line)
                self.roi_list6 = roi_list6
            elif array.GetName() == 'surf_label_ve':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    #print(roi_line)
                    self.fiber_surface_ve[lidx,0]=roi_line[0]
                    self.fiber_surface_ve[lidx, 1] = roi_line[-1]
            elif array.GetName() == 'surf_label_dk':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    #print(roi_line)
                    #print(data)
                    if not data=='HCP':
                        surf_map = numpy.load('/home/annabelchen/PycharmProjects/torch_DFC/surf_map.npy')
                        range_ctx = numpy.concatenate((numpy.array(range(1000, 1036)), numpy.array(range(2000, 2036))))
                        range_wm = numpy.concatenate((numpy.array(range(3000, 3036)), numpy.array(range(4000, 4036))))
                        range_cw = numpy.concatenate((range_ctx, range_wm))
                        if roi_line[0] in range_cw:
                            surf_label=roi_line[0]%100
                        else:
                            surf_label = surf_map[1, surf_map[0] == roi_line[0]]
                        self.fiber_surface_dk[lidx,0]=surf_label

                        if roi_line[-1] in range_cw:
                            surf_label=roi_line[-1]%100
                        else:
                            surf_label = surf_map[1, surf_map[0] == roi_line[-1]]
                        self.fiber_surface_dk[lidx,1]=surf_label
                    else:
                        self.fiber_surface_dk[lidx,0]=roi_line[0]
                        self.fiber_surface_dk[lidx, 1] = roi_line[-1]
            elif array.GetName() == 'surf_label_des':
                input_vtk_polydata.GetLines().InitTraversal()
                for lidx in range(0, input_vtk_polydata.GetNumberOfLines()):
                    #print(lidx)
                    ptids = vtk.vtkIdList()
                    input_vtk_polydata.GetLines().GetNextCell(ptids)
                    roi_line = -numpy.ones(ptids.GetNumberOfIds())
                    for pidx in range(0, ptids.GetNumberOfIds()):
                        roi_line[pidx] = array.GetTuple(ptids.GetId(pidx))[0]
                    #print(roi_line)
                    self.fiber_surface_des[lidx,0]=roi_line[0]
                    self.fiber_surface_des[lidx, 1] = roi_line[-1]


        # loop over lines
        input_vtk_polydata.GetLines().InitTraversal()
        line_ptids = vtk.vtkIdList()
        inpoints = input_vtk_polydata.GetPoints()
        vtk_array = vtk.vtkDoubleArray()
        vtk_array.SetName('Point_Seq')

        # inpd=input_vtk_polydata
        # roi = numpy.zeros((inpd.GetNumberOfLines(), 300)) - 1
        # inpointdata = inpd.GetPointData()
        # if inpointdata.GetNumberOfArrays() > 0:
        #     point_data_array_indices = list(range(inpointdata.GetNumberOfArrays()))
        #     for idx in point_data_array_indices:
        #         array = inpointdata.GetArray(idx)
        #         if array.GetName() == 'Point_ROI':
        #             inpd.GetLines().InitTraversal()
        #             for lidx in range(0, inpd.GetNumberOfLines()):
        #                 ptids = vtk.vtkIdList()
        #                 inpd.GetLines().GetNextCell(ptids)
        #                 for pidx in range(0, ptids.GetNumberOfIds()):
        #                     roi[lidx, pidx] = array.GetTuple(ptids.GetId(pidx))[0]


        for lidx in range(0, self.number_of_fibers):

            input_vtk_polydata.GetLines().GetNextCell(line_ptids)
            line_length = line_ptids.GetNumberOfIds()
            self.fiber_length[lidx]=line_length

            if self.verbose:
                if lidx % 100 == 0:
                    print("<fibers.py> Line:", lidx, "/", self.number_of_fibers)
                    print("<fibers.py> number of points:", line_length)

            for pidx in range(0, line_ptids.GetNumberOfIds()):
                vtk_array.InsertNextTuple1(pidx)
            # loop over the indices that we want and get those points
            pidx = 0
            for ind,line_index in enumerate(self._calculate_line_indices(line_length,
                                                 self.points_per_fiber)):

                # do nearest neighbor interpolation: round index
                ptidx = line_ptids.GetId(int(round(line_index)))
                #print(lidx,line_index,ptidx,line_length)
                roi=roi_list[lidx][int(round(line_index))]
                #point = inpoints.GetPoint(ptidx)
                point = list(inpoints.GetPoint(ptidx))
                point[0]=abs(point[0])
                self.fiber_array_p[lidx, pidx]=array.GetTuple(ptidx)[0]
                self.fiber_array_r[lidx, pidx] = point[0]
                self.fiber_array_a[lidx, pidx] = point[1]
                self.fiber_array_s[lidx, pidx] = point[2]
                self.fiber_roi[lidx,pidx]=roi


                if gifti is not None:
                    if ind==0 or ind==points_per_fiber-1:
                        dx = gifti[:,0] - point[0]
                        dy = gifti[:,1] - point[1]
                        dz = gifti[:,2] - point[2]
                        dx = numpy.square(dx)
                        dy = numpy.square(dy)
                        dz = numpy.square(dz)
                        distance = dx + dy + dz
                        # now compute minimum of the two endpoint distances
                        min_distance = numpy.amin(distance, 0)
                        index_min=numpy.where(distance==min_distance)
                        if ind==0:
                            self.fiber_surface[lidx, 0] = cifti[index_min]
                        else:
                            self.fiber_surface[lidx, 1] = cifti[index_min]

                if dir_flag:
                    if pidx==0:
                        #pointa = inpoints.GetPoint(ptidx + 1)
                        pointa=list(inpoints.GetPoint(ptidx+1))
                        pointa[0] = abs(pointa[0])
                        norm_diff = numpy.sqrt(numpy.sum(numpy.square(numpy.array(pointa)-numpy.array(point))))
                        self.fiber_dir_r[lidx, pidx] = (pointa[0] - point[0])/norm_diff
                        self.fiber_dir_a[lidx, pidx] = (pointa[1] - point[1])/norm_diff
                        self.fiber_dir_s[lidx, pidx] = (pointa[2] - point[2])/norm_diff
                    elif round(line_index) ==line_length-1:
                        #pointb = inpoints.GetPoint(ptidx - 1)
                        pointb = list(inpoints.GetPoint(ptidx - 1))
                        pointb[0] = abs(pointb[0])
                        norm_diff = numpy.sqrt(numpy.sum(numpy.square(numpy.array(point) - numpy.array(pointb))))
                        self.fiber_dir_r[lidx, pidx] = (point[0] - pointb[0])/norm_diff
                        self.fiber_dir_a[lidx, pidx] = (point[1] - pointb[1])/norm_diff
                        self.fiber_dir_s[lidx, pidx] = (point[2] - pointb[2])/norm_diff
                    else:
                        # pointa = inpoints.GetPoint(ptidx + 1)
                        # pointb = inpoints.GetPoint(ptidx - 1)
                        pointa = list(inpoints.GetPoint(ptidx + 1))
                        pointb = list(inpoints.GetPoint(ptidx - 1))
                        pointa[0] = abs(pointa[0])
                        pointb[0] = abs(pointb[0])
                        norm_diffa = numpy.sqrt(numpy.sum(numpy.square(numpy.array(pointa) - numpy.array(point))))
                        norm_diffb = numpy.sqrt(numpy.sum(numpy.square(numpy.array(point) - numpy.array(pointb))))
                        self.fiber_dir_r[lidx, pidx] = ((pointa[0] - point[0])/norm_diffa+(point[0] - pointb[0])/norm_diffb)/2
                        self.fiber_dir_a[lidx, pidx] = ((pointa[1] - point[1])/norm_diffa+(point[1] - pointb[1])/norm_diffb)/2
                        self.fiber_dir_s[lidx, pidx] = ((pointa[2] - point[2])/norm_diffa+(point[2] - pointb[2])/norm_diffb)/2

                pidx = pidx + 1
        input_vtk_polydata.GetPointData().AddArray(vtk_array)
        
        # initialize hemisphere info
        if self.hemispheres:
            self.calculate_hemispheres()
            
    def calculate_hemispheres(self):

        """ For each fiber assign a hemisphere using the first (R)
        coordinates.

        This part assumes we are in RAS so the first coordinate is
        positive to the RIGHT and negative to the LEFT.  The fiber
        must be more than 95% within 1 hemisphere.  This excludes
        corpus but can retain errant cingulum. We also want to
        identify likely commissural fibers.

        """

        # Figure out hemisphere of each line
        self.fiber_hemisphere = numpy.zeros(self.number_of_fibers)
        # percentage in left hemisphere
        test = sum(self.fiber_array_r.T < 0) / float(self.points_per_fiber)
        thresh = self.hemisphere_percent_threshold
        self.fiber_hemisphere[numpy.nonzero(test > thresh)] = -1
        self.fiber_hemisphere[numpy.nonzero(test < 1 - thresh)] = 1
        # previous code left for clarity below, concrete example of threshold:
        #self.fiber_hemisphere[numpy.nonzero(test > 0.95)] = -1
        #self.fiber_hemisphere[numpy.nonzero(test < 0.05)] = 1
        # otherwise hem stays 0 for commissural

        # output boolean arrays for each hemisphere and callosal fibers
        self.is_left_hem = (self.fiber_hemisphere == -1)
        self.is_right_hem = (self.fiber_hemisphere == 1)
        self.is_commissure = (self.fiber_hemisphere == 0)

        # output indices of each type above
        self.index_left_hem = numpy.nonzero(self.is_left_hem)[0]
        self.index_right_hem = numpy.nonzero(self.is_right_hem)[0]
        self.index_commissure = numpy.nonzero(self.is_commissure)[0]
        self.index_hem = \
            numpy.nonzero(self.is_left_hem | self.is_right_hem)[0]

        # output totals of each type also
        self.number_left_hem = len(self.index_left_hem)
        self.number_right_hem = len(self.index_right_hem)
        self.number_commissure = len(self.index_commissure)

        # test
        if __debug__:
            test = self.number_of_fibers == \
                (self.number_left_hem + self.number_right_hem \
                     + self.number_commissure)
            if not test:
                print("<fibers.py> ERROR: fiber numbers don't add up.")
                raise AssertionError

    def convert_to_polydata(self):
        """Convert fiber array to vtkPolyData object."""

        outpd = vtk.vtkPolyData()
        outpoints = vtk.vtkPoints()
        outlines = vtk.vtkCellArray()
        
        outlines.InitTraversal()

        for lidx in range(0, self.number_of_fibers):
            cellptids = vtk.vtkIdList()
            
            for pidx in range(0, self.points_per_fiber):

                idx = outpoints.InsertNextPoint(self.fiber_array_r[lidx, pidx],
                                                self.fiber_array_a[lidx, pidx],
                                                self.fiber_array_s[lidx, pidx])

                cellptids.InsertNextId(idx)
            
            outlines.InsertNextCell(cellptids)
            
        # put data into output polydata
        outpd.SetLines(outlines)
        outpd.SetPoints(outpoints)

        return outpd
