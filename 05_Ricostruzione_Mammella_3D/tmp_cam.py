
import bpy
cam = bpy.data.objects['Cam']
cam.location = (600.0, 60.0, 20.0)
cam.data.lens = 35
scene = bpy.context.scene
scene.render.filepath = 'C:/Users/Pierluca/Desktop/breast-mass-3d-reconstruction/05_Ricostruzione_Mammella_3D/output/P_00005/test_user_angle_fixed.png'
bpy.ops.render.render(write_still=True)
