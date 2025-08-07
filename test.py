from scitbx import math
from scitbx.array_family import flex

xyz = flex.vec3_double([(0,0,0), (1,0,0), (0,1,0)])
density = flex.double([1.0, 1.0, 1.0])

voxel_obj = math.sphere_voxel(
    15,  # np
    0,   # splat_range
    True,  # uniform
    False,  # fixed_dx
    25.0,  # external_rmax
    0.9,  # fraction
    25.0/15.0,  # dx
    xyz,
    density)

print("Voxel map size:", voxel_obj.map().size())