"""In-memory experimental short parallel gripper, independent of the source USD.

This is a geometry/constraint design comparison, not a finger-length-only ablation.
All contacts are physical box collisions; no object attachment or pose forcing.
"""
import numpy as np

PROFILE=dict(id='short_parallel',label='SHORT PARALLEL (experimental)',
             finger_length_m=.055,open_width_m=.060,travel_per_finger_m=.020,
             tcp_offset_m=.185,standoff_m=.095,revision='v3_gap60mm',command_scale=.020/.78,
             contact_static_friction=.8,contact_dynamic_friction=.65)


def build_short_parallel_gripper(stage):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
    arm_path='/Root/ur5'
    wrist_path=arm_path+'/wrist_3_link'
    old_path=arm_path+'/Robotiq_2F_140_physics_edit'
    if not stage.GetPrimAtPath(old_path):
        raise ValueError('Expected source gripper was not found')
    wrist=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(wrist_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    parent=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(arm_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    stage.RemovePrim(old_path)
    root=arm_path+'/ShortParallelGripper'
    container=UsdGeom.Xform.Define(stage,root)
    offset=np.eye(4);offset[2,3]=PROFILE['standoff_m']
    container.AddTransformOp().Set(Gf.Matrix4d((np.linalg.inv(parent)@wrist@offset).T.tolist()))
    material=UsdShade.Material.Define(stage,root+'/ContactMaterial')
    surface=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    surface.CreateStaticFrictionAttr(.8);surface.CreateDynamicFrictionAttr(.65)
    surface.CreateRestitutionAttr(0.)

    def body(name,position,dimensions,mass,color):
        path=root+'/'+name
        rigid=UsdGeom.Xform.Define(stage,path)
        rigid.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
        UsdPhysics.MassAPI.Apply(rigid.GetPrim()).CreateMassAttr(mass)
        mesh=UsdGeom.Cube.Define(stage,path+'/Box')
        mesh.CreateSizeAttr(1.)
        mesh.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
        mesh.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.001)
        mesh.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,UsdShade.Tokens.weakerThanDescendants,'physics')
        return path

    base=body('base',(0,0,.0425),(.065,.100,.035),.3,(.18,.22,.25))
    adapter=UsdGeom.Cylinder.Define(stage,base+'/RigidAdapter')
    adapter.CreateAxisAttr('Z');adapter.CreateRadiusAttr(.017);adapter.CreateHeightAttr(.105)
    adapter.AddTranslateOp().Set(Gf.Vec3d(0,0,-.0675))
    adapter.CreateDisplayColorAttr([Gf.Vec3f(.30,.35,.38)])
    UsdPhysics.CollisionAPI.Apply(adapter.GetPrim())
    left=body('left_finger',(0,-.035,.0925),(.024,.010,.055),.05,(.16,.65,.78))
    right=body('right_finger',(0,.035,.0925),(.024,.010,.055),.05,(.16,.65,.78))
    mount=UsdPhysics.FixedJoint.Define(stage,root+'/wrist_mount')
    mount.CreateBody0Rel().SetTargets([wrist_path]);mount.CreateBody1Rel().SetTargets([base])
    mount.CreateLocalPos0Attr(Gf.Vec3f(0,0,.0425+PROFILE['standoff_m']));mount.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
    joints=[]
    for name,path,y,lo,hi in [('finger_joint',left,-.035,0.,.020),
                              ('right_finger_joint',right,.035,-.020,0.)]:
        joint=UsdPhysics.PrismaticJoint.Define(stage,root+'/'+name)
        joint.CreateAxisAttr('Y')
        joint.CreateBody0Rel().SetTargets([base]);joint.CreateBody1Rel().SetTargets([path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0,y,.050))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
        joint.CreateLowerLimitAttr(lo);joint.CreateUpperLimitAttr(hi)
        joint.CreateCollisionEnabledAttr(False)
        joints.append(joint)
    drive=UsdPhysics.DriveAPI.Apply(joints[0].GetPrim(),'linear')
    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(20000.)
    drive.CreateDampingAttr(100.);drive.CreateMaxForceAttr(40.)
    drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
    # In Isaac 5 / PhysX 107, a 1-DOF prismatic mimic also uses the rotX slot.
    # This SDK ships a stale Python wrapper for this codeless schema; author
    # the same applied-schema token and fields used by the existing USD asset.
    follower=joints[1].GetPrim()
    follower.AddAppliedSchema('PhysxMimicJointAPI:rotX')
    follower.CreateRelationship('physxMimicJoint:rotX:referenceJoint').SetTargets([joints[0].GetPath()])
    for name,value in [('gearing',1.),('offset',0.),('naturalFrequency',200.),('dampingRatio',1.)]:
        follower.CreateAttribute('physxMimicJoint:rotX:'+name,Sdf.ValueTypeNames.Float).Set(value)
    return dict(PROFILE)
