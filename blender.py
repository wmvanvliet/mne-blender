"""
This is a script that you can run inside Blender. It will attempt to import
MNE-Python and load an example source estimate file. Based on the source
estimate a mesh with animated texture will be added to your scene. Be sure to
set your viewport into "shaded mode" so view the texture.

Most of all, you need to read the code below to see what it actually does and
how.

Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
"""
import bpy
import numpy as np
import matplotlib as mpl
import sys

# We need to tell Blender where to look for the MNE-Python package.
# Perhaps it's best to create a conda environment matched with the python
# version of your Blender installation. Then, install mne-base and pooch into
# it: `conda install -c conda-forge mne-base pooch`
sys.path.append('/l/vanvlm1/conda-envs/mne-blender/lib/python3.8/site-packages')
import mne
from mne.viz._3d import _process_clim, _linearize_map

root = mne.datasets.sample.data_path()
subjects_dir = f'{root}/subjects'
bem_dir = f'{subjects_dir}/sample/bem'
surf_dir = f'{subjects_dir}/sample/surf'
meg_dir = f'{root}/MEG/sample'

# Read in the mesh
surf = 'pial'
coords_lh, faces_lh = mne.read_surface(f'{surf_dir}/lh.{surf}')
coords_rh, faces_rh = mne.read_surface(f'{surf_dir}/rh.{surf}')
coords = np.vstack((coords_lh, coords_rh)).tolist()
faces = np.vstack((faces_lh, faces_rh + len(coords_lh))).tolist()
mesh = bpy.data.meshes.new(surf)
mesh.from_pydata(coords, [], faces)

# Add the mesh to the Blender shene
collection = bpy.data.collections.new('MNE-Python')
bpy.context.scene.collection.children.link(collection)
mesh_object = bpy.data.objects.new(surf, mesh)
collection.objects.link(mesh_object)

# Set up the material for the mesh
color_layer = mesh.vertex_colors.new(name='Brain Activity')
material = bpy.data.materials.new(name='Brain Activity')
material.use_nodes = True
vertex_color_node = material.node_tree.nodes.new('ShaderNodeVertexColor')
vertex_color_node.layer_name = 'Brain Activity'
bsdf_node = material.node_tree.nodes['Principled BSDF']
material.node_tree.links.new(vertex_color_node.outputs[0], bsdf_node.inputs[0])
mesh_object.data.materials.append(material)

# Smooth shading
values = [True] * len(mesh.polygons)
mesh.polygons.foreach_set("use_smooth", values)

# Read in the brain activity and convert to high resolution
stc = mne.read_source_estimate(f'{meg_dir}/sample_audvis-meg-eeg', subject='sample')
morph = mne.compute_source_morph(stc, subject_to='sample',
                                 subjects_dir=subjects_dir, spacing=None)
stc = morph.apply(stc)

# Configure the colormap
mapdata = _process_clim(clim='auto', colormap='auto', transparent=True, data=stc.data)
cmap, lims = _linearize_map(mapdata)
norm = mpl.colors.Normalize(vmin=lims[0], vmax=lims[2])
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

def my_handler(scene):
    """Invoked whenever a new frame is triggered."""
    # Determine the curretn time
    n_times = len(stc.times)
    t = 0.1 * scene.frame_current / scene.render.fps

    # We are somewhere between two samples in the data. Figure out which
    # samples and how much we should interpolate between them.
    frame_right = min(n_times - 1, np.searchsorted(stc.times, t))
    frame_left = max(0, frame_right - 1)
    interp = (t - stc.times[frame_left]) / stc.tstep
    print(f'Frame time={t}', frame_left, frame_right, interp)
    
    # Interpolate the data between frames
    interp_data = (1 - interp) * stc.data[:, frame_left]
    interp_data += interp * stc.data[:, frame_right]

    # Convert the data to colors using the colormap and assign the volors to
    # the color layer of the mesh.
    vertex_colors = mapper.to_rgba(interp_data)
    vertex_colors[:, 3] = 1 - vertex_colors[:, 3]
    face_colors = vertex_colors[faces]
    color_layer.data.foreach_set('color', face_colors.ravel())
    mesh.update() 

bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(my_handler)
my_handler(bpy.data.scenes[0])
