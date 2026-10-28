"""Physical dish and raised cup stand; marker IDs identify their support surfaces."""
import math
from pathlib import Path
import cv2,numpy as np

def build_destinations(stage):
    from pxr import UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
    root='/Root/Destinations';UsdGeom.Xform.Define(stage,root)
    folder=Path('/experiment/assets/destination_markers');folder.mkdir(parents=True,exist_ok=True)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    def solid(name,xyz,size,color,cylinder=False):
        shape=(UsdGeom.Cylinder.Define(stage,root+'/'+name) if cylinder else UsdGeom.Cube.Define(stage,root+'/'+name))
        if cylinder:shape.CreateRadiusAttr(size[0]);shape.CreateHeightAttr(size[2]);shape.CreateAxisAttr('Z')
        else:shape.CreateSizeAttr(1.);shape.AddScaleOp().Set(Gf.Vec3f(*size))
        shape.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        # Translate before scale in the authored transform order.
        if not cylinder:shape.SetXformOpOrder(list(reversed(shape.GetOrderedXformOps())))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*color)]);UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        return shape
    fixtures={10:dict(name='plate',center=[.33,-.27,.018],radius=.067),20:dict(name='cup_stand',center=[.56,-.28,.070],radius=.055)}
    solid('PlateBase',(.33,-.27,.014),(.071,0,.008),(.93,.94,.96),True)
    for i in range(32):
        angle=2*math.pi*i/32
        shape=solid('Rim'+str(i),(.33+.073*math.cos(angle),-.27+.073*math.sin(angle),.022),(.015,.006,.016),(.15,.35,.70))
        shape.AddRotateZOp().Set(math.degrees(angle)+90)
        ops=shape.GetOrderedXformOps();shape.SetXformOpOrder([ops[0],ops[2],ops[1]])
    solid('StandTop',(.56,-.28,.064),(.13,.13,.012),(.45,.25,.10))
    for i,(dx,dy) in enumerate([(-.05,-.05),(.05,-.05),(-.05,.05),(.05,.05)]):
        solid('Leg'+str(i),(.56+dx,-.28+dy,.034),(.012,.012,.048),(.28,.16,.08))
    for ident,spec in fixtures.items():
        marker=cv2.aruco.generateImageMarker(dictionary,ident,400)
        marker=cv2.copyMakeBorder(marker,50,50,50,50,cv2.BORDER_CONSTANT,value=255)
        texture=folder/f'{ident}.png';cv2.imwrite(str(texture),marker)
        path=root+'/Marker'+str(ident);mesh=UsdGeom.Mesh.Define(stage,path)
        x,y,z=spec['center'];s=.05
        mesh.CreatePointsAttr([Gf.Vec3f(x-s,y-s,z+.0002),Gf.Vec3f(x+s,y-s,z+.0002),Gf.Vec3f(x+s,y+s,z+.0002),Gf.Vec3f(x-s,y+s,z+.0002)])
        mesh.CreateFaceVertexCountsAttr([4]);mesh.CreateFaceVertexIndicesAttr([0,1,2,3]);mesh.CreateSubdivisionSchemeAttr('none')
        uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex);uv.Set([(0,0),(1,0),(1,1),(0,1)])
        mat=UsdShade.Material.Define(stage,path+'/Material');shader=UsdShade.Shader.Define(stage,path+'/Material/Surface');shader.CreateIdAttr('UsdPreviewSurface')
        tex=UsdShade.Shader.Define(stage,path+'/Material/Texture');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(texture))
        reader=UsdShade.Shader.Define(stage,path+'/Material/UV');reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(),'rgb')
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return fixtures
