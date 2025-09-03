import sys

from pymol import cmd

def load (ref, map):
    cmd.load(ref,'ref')
    cmd.load(map, 'map')
    cmd.show_as('spheres','map')
    cmd.show_as('surface', 'ref')
    cmd.color('magenta', 'ref')
    cmd.color('marine','map')
    cmd.set('sphere_transparency', 0.5, 'map')


def make_image(out_file):
    cmd.bg_colour('white')
    cmd.set('ray_trace_mode',0)
    cmd.set('antialias', 2)
    cmd.set("shininess", 3)
    cmd.set("orthoscopic", "on")
    cmd.ray()
    out_file = out_file.replace('.png', '_1.png')
    cmd.png(out_file, dpi=300)
    cmd.rotate('y', 90)
    cmd.ray()
    out_file = out_file.replace('_1.png', '_2.png')
    cmd.png(out_file + '', dpi=300)
    cmd.rotate('x', 270)
    cmd.ray()
    out_file = out_file.replace('_2.png', '_3.png')
    cmd.png(out_file+'', dpi=300)



map_pdb_path = sys.argv[1]
target_pdb_path = sys.argv[2]
load(target_pdb_path, map_pdb_path)
out_file = sys.argv[3]
make_image(out_file)