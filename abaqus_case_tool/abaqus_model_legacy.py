#!/usr/bin/python
# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
import regionToolset
import mesh
job_path = 'D:/SIMULIA/temp/'
file_path = 'D:/UPython/abaqus-case/'
num_cpus = 4


class IntermediateCase:

    def __init__(self, i=3.0, s=3.5, o=4.0, p=2.5, sub_job=False, job_name='Job-case-1'):
        # 参数
        self.job_name = job_name
        self.th_inner = i  # 内机匣厚度
        self.th_split = s  # 分流环厚度
        self.th_outer = o  # 外机匣厚度
        self.th_plate = p  # 支板厚度
        self.w_plate = 50.0  # 支板外轮廓宽度
        self.w_mount = 40.0  # 外机匣加强筋宽度
        self.h_mount = 765.0  # 外机匣加强筋高度
        self.mass = None  # 机匣质量
        self.m = None  # mdb.models['Model-1']
        self.a = None  # mdb.models['Model-1'].rootAssembly
        self.p = None  # mdb.models['Model-1'].parts['case']

        # 建模
        self.set_model()
        self.set_material()
        self.set_couple()
        self.set_load()
        self.set_mesh()
        self.set_job(sub_job)

    @staticmethod
    def __sketch_lines(sketch, points):
        point_sum = len(points)
        if point_sum<=1:
            return None
        for i in range(point_sum-1):
            next_0 = points[i+1][0]
            if type(next_0)==str:
                prev_x, prev_y = points[i][0], points[i][1]
                dist = points[i+1][1]
                if next_0=='↑':
                    points[i+1] = (prev_x, prev_y+dist)
                elif next_0=='↓':
                    points[i+1] = (prev_x, prev_y-dist)
                elif next_0=='←':
                    points[i+1] = (prev_x-dist, prev_y)
                elif next_0=='→':
                    points[i+1] = (prev_x+dist, prev_y)
                else:
                    print('MULTILINE ERROR!!!')
                    return None
            sketch.Line(point1=points[i], point2=points[i+1])
        sketch.Line(point1=points[-1], point2=points[0])

    def __part_revolved(self, part_name, *lines):
        s = self.m.ConstrainedSketch(name=part_name, sheetSize=1600.0)
        l = s.ConstructionLine(point1=(0.0, 0.0), point2=(100.0, 0.0))
        s.assignCenterline(line=l)
        for line_points in lines:
            self.__sketch_lines(s, line_points)
        p = self.m.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseSolidRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
        # del self.m.sketches['__skt__']

    def __part_cases(self, part_name='cases'):
        ht_outer = 39.0-self.th_outer
        points_inner_case = [(0.0, 187.0), ('→', 12.0), ('↑', 5.0), ('←', 9.0), ('↑', 12.0-self.th_inner),
                             ('→', 339.5), ('↓', 17.0-self.th_inner), ('→', 103.0), ('↓', 55.0), ('→', 4.5),
                             ('↑', 17.0), ('→', 6.0), ('↑', 29.0), ('→', 8.0), ('↑', 26.0), ('←', 464.0), ('↓', 17.0)]
        points_split_case = [(-175.0, 485.0), ('→', 275.0), (349.5, 463.5), ('↑', 4.5), (200.0, 480.0),
                             (200.0, 485.0), (349.5, 485.0), ('↑', self.th_split), ('←', 524.5), ('↓', self.th_split)]
        points_outer_case = [(-120.5, 728.0), ('→', 470.0), ('↑', 39.0), ('←', 12.0), ('↓', ht_outer),
                             ('←', 59.0), ('↑', ht_outer), ('←', 16), ('↓', ht_outer),
                             ('←', 59.0), ('↑', ht_outer), ('←', 16), ('↓', ht_outer),
                             ('←', 59.0), ('↑', ht_outer), ('←', 16), ('↓', ht_outer),
                             ('←', 59.0), ('↑', ht_outer), ('←', 16), ('↓', ht_outer),
                             (-113.5, 728.0+self.th_outer), (-113.5, 770.0), ('←', 7.0)]
        self.__part_revolved(part_name, points_inner_case, points_split_case, points_outer_case)

    def __part_plate(self, part_name='plate', th=0.0):
        x_1, x_2, y = 100.0-th*2, 200.0-th*2, self.w_plate*0.5-th
        s = self.m.ConstrainedSketch(name='__skt__', sheetSize=800.0)
        g = s.geometry
        s.EllipseByCenterPerimeter(center=(125.0, 0.0), axisPoint1=(125.0, y), axisPoint2=(125.0-x_1, 0.0))
        s.EllipseByCenterPerimeter(center=(125.0, 0.0), axisPoint1=(125.0, y), axisPoint2=(125.0+x_2, 0.0))
        s.autoTrimCurve(curve1=g.findAt((125.0+x_1, 0.0)), point1=(125.0+x_1, 0.0))
        s.autoTrimCurve(curve1=g.findAt((125.0-x_2, 0.0)), point1=(125.0-x_2, 0.0))
        p = self.m.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseSolidExtrude(sketch=s, depth=800.0)
        del self.m.sketches['__skt__']

    def __part_plate_bound(self, part_name='plate_bound'):
        points_inner_boundary = [(0.0, 0.0), (500.0, 0.0), (500.0, 204.0), (0.0, 204.0)]
        points_outer_boundary = [(0.0, 728.0), (500.0, 728.0), (500.0, 1000.0), (0.0, 1000.0)]
        self.__part_revolved(part_name, points_inner_boundary, points_outer_boundary)

    def __part_mount(self, part_name='mount', width=40.0):
        w = width*0.5
        s = self.m.ConstrainedSketch(name='__skt__', sheetSize=800.0)
        points_mount = [(-113.5, -w), (-113.5, w), (337.5, w), (337.5, -w)]
        self.__sketch_lines(s, points_mount)
        p = self.m.Part(name=part_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
        p.BaseSolidExtrude(sketch=s, depth=800.0)
        del self.m.sketches['__skt__']

    def __part_mount_bound(self, part_name='mount_bound'):
        points_boundary = [(-200, 0.0), (500.0, 0.0), (500.0, 728.0), (-200.0, 728.0)]
        self.__part_revolved(part_name, points_boundary)

    def set_model(self):
        Mdb()
        self.m = mdb.models['Model-1']
        self.a = mdb.models['Model-1'].rootAssembly
        self.__part_cases('cases')
        self.__part_plate('plate_outer', 0.0)
        self.__part_plate('plate_inner', self.th_plate)
        self.__part_plate_bound('plate_bound')
        self.__part_mount('mount', self.w_mount)
        self.__part_mount('mount_bound_upper', 100.0)
        self.__part_mount_bound('mount_bound_lower')
        self.a.Instance(name='cases', part=self.m.parts['cases'], dependent=OFF)
        self.a.Instance(name='plate_outer', part=self.m.parts['plate_outer'], dependent=OFF)
        self.a.Instance(name='plate_inner', part=self.m.parts['plate_inner'], dependent=OFF)
        self.a.Instance(name='plate_bound', part=self.m.parts['plate_bound'], dependent=OFF)
        self.a.Instance(name='mount', part=self.m.parts['mount'], dependent=OFF)
        self.a.Instance(name='mount_bound_u', part=self.m.parts['mount_bound_upper'], dependent=OFF)
        self.a.Instance(name='mount_bound_l', part=self.m.parts['mount_bound_lower'], dependent=OFF)
        self.a.InstanceFromBooleanCut(name='plate_o', instanceToBeCut=self.a.instances['plate_outer'],
                                      cuttingInstances=(self.a.instances['plate_bound'],), originalInstances=SUPPRESS)
        self.a.InstanceFromBooleanCut(name='plate_fix', instanceToBeCut=self.a.instances['plate_o-1'],
                                      cuttingInstances=(self.a.instances['plate_inner'],), originalInstances=SUPPRESS)
        self.a.features['plate_bound'].resume()
        self.a.features['plate_inner'].resume()
        self.a.InstanceFromBooleanCut(name='plate_i', instanceToBeCut=self.a.instances['plate_inner'],
                                      cuttingInstances=(self.a.instances['plate_bound'],), originalInstances=SUPPRESS)
        self.a.InstanceFromBooleanCut(name='mount_fix', instanceToBeCut=self.a.instances['mount'],
                                      cuttingInstances=(self.a.instances['mount_bound_l'],), originalInstances=SUPPRESS)
        self.a.translate(instanceList=('mount_bound_u', ), vector=(0.0, 0.0, self.h_mount))
        self.a.RadialInstancePattern(instanceList=('mount_fix-1', ), point=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                                     number=8, totalAngle=360.0)
        self.a.RadialInstancePattern(instanceList=('mount_bound_u', ), point=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                                     number=8, totalAngle=360.0)
        self.a.RadialInstancePattern(instanceList=('plate_fix-1', ), point=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                                     number=8, totalAngle=360.0)
        self.a.RadialInstancePattern(instanceList=('plate_i-1', ), point=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0),
                                     number=8, totalAngle=360.0)
        instances_plate_i = [self.a.instances['plate_i-1']]
        for i in range(2, 9):
            instances_plate_i.append(self.a.instances['plate_i-1-rad-%g'%i])
        self.a.InstanceFromBooleanCut(name='case_temp1', instanceToBeCut=self.a.instances['cases'],
                                      cuttingInstances=instances_plate_i, originalInstances=SUPPRESS)
        instances_fix = [self.a.instances['case_temp1-1'],
                         self.a.instances['plate_fix-1'],
                         self.a.instances['mount_fix-1']]
        for i in range(2, 9):
            instances_fix.append(self.a.instances['plate_fix-1-rad-%g'%i])
            instances_fix.append(self.a.instances['mount_fix-1-rad-%g'%i])
        self.a.InstanceFromBooleanMerge(name='case_temp2', instances=instances_fix,
                                        originalInstances=SUPPRESS, domain=GEOMETRY)
        instances_bound_u = [self.a.instances['mount_bound_u']]
        for i in range(2, 9):
            instances_bound_u.append(self.a.instances['mount_bound_u-rad-%g'%i])
        self.a.InstanceFromBooleanCut(name='case', instanceToBeCut=self.a.instances['case_temp2-1'],
                                      cuttingInstances=instances_bound_u, originalInstances=SUPPRESS)
        for key in self.a.instances.keys():
            del self.a.instances[key]
        for key in [key for key in self.m.parts.keys() if key!='case']:
            del self.m.parts[key]
        self.p = self.m.parts['case']
        e, v, d, c = self.p.edges, self.p.vertices, self.p.datums, self.p.cells
        # 倒角
        r_io, r_io2 = 204.0, 204.0*0.7071067811865476
        edge_list = (
            e.getClosest(((25.0, r_io, 1.0),), )[0][0], e.getClosest(((25.0, r_io, -1.0),), )[0][0],
            e.getClosest(((325.0, r_io, 1.0),), )[0][0], e.getClosest(((325.0, r_io, -1.0),), )[0][0],
            e.findAt(coordinates=(25.0, -r_io, 0.0)), e.findAt(coordinates=(325.0, -r_io, 0.0)),
            e.findAt(coordinates=(25.0, 0.0, r_io)), e.findAt(coordinates=(325.0, 0.0, r_io)),
            e.findAt(coordinates=(25.0, 0.0, -r_io)), e.findAt(coordinates=(325.0, 0.0, -r_io)),
            e.findAt(coordinates=(25.0, r_io2, r_io2)), e.findAt(coordinates=(325.0, r_io2, r_io2)),
            e.findAt(coordinates=(25.0, r_io2, -r_io2)), e.findAt(coordinates=(325.0, r_io2, -r_io2)),
            e.findAt(coordinates=(25.0, -r_io2, r_io2)), e.findAt(coordinates=(325.0, -r_io2, r_io2)),
            e.findAt(coordinates=(25.0, -r_io2, -r_io2)), e.findAt(coordinates=(325.0, -r_io2, -r_io2)),
        )
        self.p.Round(radius=15.0, edgeList=edge_list)
        # 切割
        cells = c.getByBoundingBox(-800,-800,-800,800,800,800)
        edges = (e.findAt(coordinates=(349.5, 0.0, 728.0)),
                 e.findAt(coordinates=(349.5, 0.0, 485.0+self.th_split)),
                 e.findAt(coordinates=(349.5, 0.0, 485.0)),)
        self.p.PartitionCellByExtrudeEdge(line=e.findAt(coordinates=(370.0, 204.0, 0.0)),
                                          cells=cells, edges=edges, sense=REVERSE)
        cells = c.getByBoundingBox(-800,-800,-800,800,800,800)
        edges = (e.findAt(coordinates=(464.0, 0.0, 204.0)),)
        self.p.PartitionCellByExtrudeEdge(line=e.findAt(coordinates=(370.0, 204.0, 0.0)),
                                          cells=cells, edges=edges, sense=REVERSE)
        cells = c.findAt(((349.0, 0.0, 464.0), ))
        self.p.PartitionCellByPlaneThreePoints(point1=v.getClosest(((200.0, -self.w_plate*0.5, 485.0),), )[0][0],
                                               point2=v.getClosest(((200.0, self.w_plate*0.5, 485.0),), )[0][0],
                                               point3=v.getClosest(((200.0, self.w_plate*0.5, 480.0),), )[0][0],
                                               cells=cells)
        cells = c.findAt(((200.0, 0.0, self.h_mount-1.0), ))
        self.p.PartitionCellByPlanePointNormal(point=v.findAt(coordinates=(278.5, 20.0, self.h_mount)),
                                               normal=e.findAt(coordinates=(450.0, 204.0, 0.0)), cells=cells)
        cells = c.findAt(((200.0, 0.0, self.h_mount-1.0), ))
        self.p.PartitionCellByPlanePointNormal(point=v.findAt(coordinates=(37.5, 20.0, self.h_mount)),
                                               normal=e.findAt(coordinates=(450.0, 204.0, 0.0)), cells=cells)
        # 定义
        self.a.Instance(name='case', part=self.p, dependent=OFF)
        session.viewports['Viewport: 1'].setValues(displayedObject=self.a)

    def set_material(self):
        self.m.Material('Ti65')
        self.m.materials['Ti65'].Elastic(table=((121000.0, 0.31,), ))
        self.m.materials['Ti65'].Density(table=((4.59e-09, ), ))
        self.m.HomogeneousSolidSection(name='Ti65', material='Ti65', thickness=None)
        region = regionToolset.Region(cells=self.p.cells.getByBoundingBox(-800,-800,-800,800,800,800))
        self.p.SectionAssignment(region=region, sectionName='Ti65', offset=0.0,
                                 offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        self.mass = self.p.getMassProperties(self.p.cells.getByBoundingBox(-800,-800,-800,800,800,800))['mass']

    def set_couple(self):
        rp = self.a.referencePoints
        s = self.a.instances['case'].faces
        self.a.ReferencePoint(point=(-400.0, 0.0, 0.0))
        self.a.ReferencePoint(point=(600.0, 0.0, 0.0))
        self.a.ReferencePoint(point=(700.0, 0.0, 0.0))
        self.a.ReferencePoint(point=(800.0, 0.0, 0.0))
        region_H_p = self.a.Set(referencePoints=(rp.findAt((-400.0, 0.0, 0.0)),), name='RP-H')
        region_H_f = regionToolset.Region(side1Faces=s.findAt(((-120.5, 0.0, 730.0),),))
        region_F_p = self.a.Set(referencePoints=(rp.findAt((600.0, 0.0, 0.0)),), name='RP-F')
        region_F_f = regionToolset.Region(side1Faces=s.findAt(((450.0, 0.0, 135.0),),))
        region_M_p = self.a.Set(referencePoints=(rp.findAt((700.0, 0.0, 0.0)),), name='RP-M')
        region_M_f = regionToolset.Region(side1Faces=s.findAt(((349.5, 0.0, 465.0),), ((349.5, 0.0, 486.0),),))
        region_V_p = self.a.Set(referencePoints=(rp.findAt((800.0, 0.0, 0.0)),), name='RP-V')
        region_V_f = regionToolset.Region(side1Faces=s.findAt(((349.5, 0.0, 745.0),),))
        self.m.Coupling(name='Constraint-H', controlPoint=region_H_p, surface=region_H_f,
                        influenceRadius=WHOLE_SURFACE, couplingType=STRUCTURAL, weightingMethod=UNIFORM,
                        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        self.m.Coupling(name='Constraint-F', controlPoint=region_F_p, surface=region_F_f,
                        influenceRadius=WHOLE_SURFACE, couplingType=STRUCTURAL, weightingMethod=UNIFORM,
                        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        self.m.Coupling(name='Constraint-M', controlPoint=region_M_p, surface=region_M_f,
                        influenceRadius=WHOLE_SURFACE, couplingType=STRUCTURAL, weightingMethod=UNIFORM,
                        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
        self.m.Coupling(name='Constraint-V', controlPoint=region_V_p, surface=region_V_f,
                        influenceRadius=WHOLE_SURFACE, couplingType=STRUCTURAL, weightingMethod=UNIFORM,
                        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)

    def set_load(self):
        s = self.a.instances['case'].faces
        self.m.StaticStep(name='Step-1', previous='Initial',)
        # self.m.Moment(name='T-V', createStepName='Step-1', region=self.a.sets['RP-V'], cm1=19947.0,
        #               distributionType=UNIFORM, field='', localCsys=None)
        # self.m.Moment(name='T-M', createStepName='Step-1', region=self.a.sets['RP-M'], cm1=-22640.0,
        #               distributionType=UNIFORM, field='', localCsys=None)
        # self.m.ConcentratedForce(name='Load-H', createStepName='Step-1', region=self.a.sets['RP-H'], cf1=-1956.0,
        #                          distributionType=UNIFORM, field='',localCsys=None)
        # self.m.ConcentratedForce(name='Load-V', createStepName='Step-1', region=self.a.sets['RP-V'], cf1=141502.0,
        #                          distributionType=UNIFORM, field='',localCsys=None)
        # self.m.ConcentratedForce(name='Load-M', createStepName='Step-1', region=self.a.sets['RP-M'], cf1=-237014.0,
        #                          distributionType=UNIFORM, field='',localCsys=None)
        self.m.ConcentratedForce(name='Load-F', createStepName='Step-1', region=self.a.sets['RP-F'], cf1=-45516.0,
                                 distributionType=UNIFORM, field='',localCsys=None)
        self.a.Set(faces=s.findAt(((250.0, 0.0, self.h_mount),),), name='FACE-V')
        self.m.YsymmBC(name='BC-V', createStepName='Initial', region=self.a.sets['FACE-V'], localCsys=None)
        self.a.Set(faces=s.findAt(((250.0, -self.h_mount, 0.0),), ((250.0, self.h_mount, 0.0),),), name='FACE-H')
        self.m.YasymmBC(name='BC-H', createStepName='Initial', region=self.a.sets['FACE-H'], localCsys=None)

    def set_mesh(self):
        x_0, x_1 = 25.0-15.0, 325.0+15.0
        r_io, r_io2 = 204.0, 204.0*0.7071067811865476
        e, c = self.a.instances['case'].edges, self.a.instances['case'].cells
        cells = c.getByBoundingBox(-800,-800,-800,800,800,800)
        self.a.setMeshControls(regions=cells, elemShape=TET, technique=FREE)
        elemType1 = mesh.ElemType(elemCode=C3D20R)
        elemType2 = mesh.ElemType(elemCode=C3D15)
        elemType3 = mesh.ElemType(elemCode=C3D10)
        self.a.setElementType(regions=(cells, ), elemTypes=(elemType1, elemType2, elemType3))
        partInstances =(self.a.instances['case'], )
        edges = (
            e.getClosest(((x_0, r_io, 1.0),), )[0][0], e.getClosest(((x_0, r_io, -1.0),), )[0][0],
            e.getClosest(((x_1, r_io, 1.0),), )[0][0], e.getClosest(((x_1, r_io, -1.0),), )[0][0],
            e.findAt(coordinates=(x_0, -r_io, 0.0)), e.findAt(coordinates=(x_1, -r_io, 0.0)),
            e.findAt(coordinates=(x_0, 0.0, r_io)), e.findAt(coordinates=(x_1, 0.0, r_io)),
            e.findAt(coordinates=(x_0, 0.0, -r_io)), e.findAt(coordinates=(x_1, 0.0, -r_io)),
            e.findAt(coordinates=(x_0, r_io2, r_io2)), e.findAt(coordinates=(x_1, r_io2, r_io2)),
            e.findAt(coordinates=(x_0, r_io2, -r_io2)), e.findAt(coordinates=(x_1, r_io2, -r_io2)),
            e.findAt(coordinates=(x_0, -r_io2, r_io2)), e.findAt(coordinates=(x_1, -r_io2, r_io2)),
            e.findAt(coordinates=(x_0, -r_io2, -r_io2)), e.findAt(coordinates=(x_1, -r_io2, -r_io2)),
            e.findAt(coordinates=(250.0, 204.0-self.th_inner, 0.0))
        )
        self.a.seedEdgeBySize(edges=edges, size=15.0, deviationFactor=0.1, constraint=FINER)
        self.a.seedPartInstance(regions=partInstances, size=18.0, deviationFactor=0.1, minSizeFactor=0.1)
        self.a.generateMesh(regions=c.getByBoundingBox(0, -485, -485, 464, 485, 485))
        self.a.generateMesh(regions=c.getByBoundingBox(-800,-800,-800,800,800,800))

    def set_job(self, sub_job=False):
        mdb.Job(name=self.job_name, model='Model-1', description='', type=ANALYSIS,
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=num_cpus,
                numDomains=num_cpus, numGPUs=0)
        if sub_job:
            mdb.jobs[self.job_name].submit(consistencyChecking=OFF)
            mdb.jobs[self.job_name].waitForCompletion()


class PostProcess:

    def __init__(self, job_name='Job-case-1'):
        # 参数
        self.job_path = job_path
        self.job_name = job_name
        self.view = session.viewports['Viewport: 1']

        # 操作
        self.result = self.get_result()

    def __open_odb(self):
        odb = session.openOdb(name=self.job_path+self.job_name+'.odb')
        self.view.setValues(displayedObject=odb)
        self.view.odbDisplay.display.setValues(plotState=(CONTOURS_ON_UNDEF,))

    def __set_frame(self, step=0, frame=1):
        self.view.odbDisplay.setFrame(step=step, frame=frame)

    def __set_value(self, data_type=('S', INVARIANT, 'Mises'), position_type=INTEGRATION_POINT):
        self.view.odbDisplay.setPrimaryVariable(
            variableLabel=data_type[0], outputPosition=position_type, refinement=(data_type[1], data_type[2]))

    def __get_value_max(self):
        return self.view.odbDisplay.contourOptions.autoMaxValue

    def get_result(self):
        self.__open_odb()
        self.__set_frame()
        self.__set_value(('S', INVARIANT, 'Mises'),INTEGRATION_POINT)
        mises = self.__get_value_max()
        self.__set_value(('U', COMPONENT, 'U2'), NODAL)
        stiff_y = 45516000/self.__get_value_max()
        self.__set_value(('U', COMPONENT, 'U3'), NODAL)
        stiff_z = 45516000/self.__get_value_max()
        return mises, stiff_y, stiff_z


def create_case_data(start=(0, 0, 0, 0), file_name='result'):
    print('\n========= start !!! ==========')
    range_i = range_f(3.0, 7.0, 1.0)
    range_s = range_f(3.0, 9.0, 1.0)
    range_o = range_f(4.0, 10.0, 1.0)
    range_p = range_f(2.0, 6.0, 1.0)
    errors = []
    file = open(file_path+file_name+'.txt', 'a')
    for i in range_i:
        if i<start[0]:
            continue
        for s in range_s:
            if (i==start[0])and(s<start[1]):
                continue
            for o in range_o:
                if (i==start[0])and(s==start[1])and(o<start[2]):
                    continue
                for p in range_p:
                    if (i==start[0])and(s==start[1])and(o==start[2])and(p<start[3]):
                        continue
                    try:
                        print("\n%g    %g    %g    %g"%(i, s, o, p))
                        mass = IntermediateCase(i, s, o, p, True).mass
                        mises, stiff_y, stiff_z = PostProcess().result
                        for datum in (i, s, o, p, mass, mises, stiff_y, stiff_z):
                            file.write("%g    "%datum)
                        file.write("\n")
                    except BaseException as exp:
                        file.write('Error!!!\n')
                        errors.append([i, s, o, p, exp])
    for error in errors:
        file.write(str(error))
        file.write('\n')
    file.close()
    print('========= finish !!! ==========\n')


def range_f(start, end, interval):
    range_ = []
    while start<=end:
        range_.append(start)
        start = round(start+interval, 3)
    return range_


if __name__=='__main__':
    # IntermediateCase(3.0, 3.5, 4.0, 2.5, False)
    # IntermediateCase(7.0, 9.0, 10.0, 6.0, False)
    # IntermediateCase(10, 10, 10, 6, False)
    # IntermediateCase(8.9, 6.2, 9.9, 6.1, True)
    print(PostProcess().result)

    # create_case_data()
