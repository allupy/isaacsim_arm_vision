"""Textured YCB objects, authored only into the experimental stage.

Sources: YCB benchmark Google 16k meshes; no kinematic attachments or grasp cheats.
The scale is reduced to fit the 60 mm experimental parallel gripper.
"""
from pathlib import Path
import json
import numpy as np


def add_everyday_scene(stage):
    from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt
    for name in ('red_cube','green_cube','blue_cube'):
        stage.RemovePrim('/Root/'+name)
    specs = [('apple','013_apple',(.52,-.17),.048,.07),
             ('banana','011_banana',(.64,-.01),.130,.06),
             ('cup','025_mug',(.51,.14),.068,.08)]
    metadata = {}
    for name, asset, xy, longest, mass in specs:
        folder = Path('/experiment/assets/ycb')/asset/'google_16k'
        vertices=[]; uv=[]; indices=[]; uv_indices=[]; counts=[]
        for line in (folder/'textured.obj').read_text().splitlines():
            fields=line.split()
            if not fields: continue
            if fields[0]=='v': vertices.append([float(x) for x in fields[1:4]])
            elif fields[0]=='vt': uv.append([float(x) for x in fields[1:3]])
            elif fields[0]=='f':
                counts.append(len(fields)-1)
                for value in fields[1:]:
                    parts=value.split('/')
                    indices.append(int(parts[0])-1)
                    uv_indices.append(int(parts[1])-1)
        points=np.asarray(vertices)
        # Banana's long axis and mug's handle axis lie along gripper finger X.
        if name in ('banana','cup'):
            _,vectors=np.linalg.eigh(np.cov(points[:,:2].T))
            long=vectors[:,-1]
            angle=-np.arctan2(long[1],long[0])
            rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            points[:,:2]=points[:,:2]@rotation.T
        low,high=points.min(0),points.max(0)
        points=(points-(low+high)/2)*longest/max(high-low)
        size=np.ptp(points,axis=0)
        path='/Root/'+name
        root=UsdGeom.Xform.Define(stage,path)
        root.AddTranslateOp().Set(Gf.Vec3d(*xy,float(.0105+size[2]/2)))
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
        mesh=UsdGeom.Mesh.Define(stage,path+'/Geometry')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr(counts);mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        st=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv,dtype=np.float32)))
        st.SetIndices(uv_indices)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        mesh.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
        mesh.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.001)
        mesh.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
        material=UsdShade.Material.Define(stage,path+'/Material')
        surf=UsdShade.Shader.Define(stage,path+'/Material/Surface')
        surf.CreateIdAttr('UsdPreviewSurface')
        surf.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        texture=UsdShade.Shader.Define(stage,path+'/Material/Texture')
        texture.CreateIdAttr('UsdUVTexture')
        texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(folder/'texture_map.png'))
        reader=UsdShade.Shader.Define(stage,path+'/Material/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        surf.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(),'rgb')
        material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(),'surface')
        friction=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        friction.CreateStaticFrictionAttr(.8);friction.CreateDynamicFrictionAttr(.65);friction.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
        metadata[name]=dict(source=asset,size_m=size.tolist(),mass_kg=mass,collision='convexHull')
    print('DEMO_OBJECTS '+json.dumps(metadata),flush=True)
    return list(metadata)
