# coding=utf-8
import evo.main_ape as eape
import evo.main_rpe as erpe
from evo.core import metrics,trajectory
from evo.core import sync
from evo.tools import file_interface
from evo.core.metrics import PoseRelation, Unit
import os
"""
全局轨迹相似度，使用APE进行度量，先进行align操作：full_transformation
对局部trans和rotation误差，使用RPE进行度量，需要设置度量的窗口（一般以m为单位），且不进行align操作：'translation_part','rotation_part'
"""
import numpy as np
#评估模式 全部/平移部分/旋转部分/旋转部分-弧度/旋转部分-角度
pose_relation=['full_transformation','translation_part','rotation_part','rotation_angle_rad','rotation_angle_deg']

#基于kitti的PosePath3D和时间戳times序列构建PoseTrajectory3D数据结构，用于同步
def path2trajectory(rpose,rtime):
    rtraj=trajectory.PoseTrajectory3D(positions_xyz=rpose.positions_xyz,orientations_quat_wxyz=rpose.orientations_quat_wxyz,timestamps=rtime, poses_se3=rpose.poses_se3)
    return rtraj

def get_ape_static_raw(data,mode):
    ape_metric = metrics.APE(mode)
    ape_metric.process_data(data)
    ape_stat = np.asarray(list(ape_metric.get_all_statistics().values()))#总体误差统计值
    error_full=ape_metric.error#单个误差
    return ape_stat,error_full

def get_rpe_static_raw(data,mode,delta,unit,allpair):
    rpe_metric = metrics.RPE(mode,delta,unit,allpair)#
    rpe_metric.process_data(data)
    rpe_stat = np.asarray(list(rpe_metric.get_all_statistics().values()))
    error_full=rpe_metric.error
    return rpe_stat,error_full

#参数文件路径@GT—pose@GT-time@EST-pose@EST-time
def eval(ref_p_path,ref_t_path,est_p_path,est_t_path):
    #pose1
    ref_pose=file_interface.read_kitti_poses_file(ref_p_path)#PosePath3D
    ref_time=np.loadtxt(ref_t_path)#time
    assert len(ref_time)==ref_pose.num_poses
    ref_traj=path2trajectory(ref_pose,ref_time)#PoseTrajectory3D
    #pose2
    est_pose=file_interface.read_kitti_poses_file(est_p_path)#PosePath3D
    est_time=np.loadtxt(est_t_path)#time
    assert len(est_time)==est_pose.num_poses
    est_traj=path2trajectory(est_pose,est_time)#PoseTrajectory3D
    #print(ref_traj.get_infos(),est_traj.get_infos())
    #sync-基于少量数据同步
    max_diff = 0.1 # 雷达10Hz， 最大为
    traj_ref, traj_est = sync.associate_trajectories(ref_traj, est_traj, max_diff)

    data=(traj_ref,traj_est)

    #APE--全局评估
    ape_pose_relation = metrics.PoseRelation.full_transformation
    ape_statis, ape_err=get_ape_static_raw(data,ape_pose_relation)
    ape_err=np.asarray(ape_err)
    #print(ape_statis,ape_total)
    #RPE-细节评估
    rpe_relation_t = metrics.PoseRelation.translation_part
    rpe_relation_r = metrics.PoseRelation.rotation_part
    """
    frame 一般不涉及单位，可以进行逐帧对比 取值delta=1，不太影响评估误差数量
    使用frame，1，利用最小粒度单位进行详细评估，寻找异常帧位置，从而修改代码。
    meters 可以通过设置10/20/50/100 m ，会缩小评估误差数量
    degrees=radians, filter_pairs_by_angle
    """
    delta=20 #评估窗口
    delta_unit=metrics.Unit.meters #评估单位 [meters,frames,degrees]
    all_pairs = False
    rpe_statis_t,rpe_trans=get_rpe_static_raw(data,rpe_relation_t,delta,delta_unit,all_pairs)#translation
    rpe_statis_r,rpe_rotat=get_rpe_static_raw(data,rpe_relation_r,delta,delta_unit,all_pairs)#rotation
    #
    static=np.vstack((ape_statis,rpe_statis_t,rpe_statis_r))# 固定统计类，可合并
    rpe_err=np.vstack((rpe_trans,rpe_rotat))#数据不等长，不可合并
    return static,ape_err,rpe_err

def save_results(ppath,mode,static,ape_error,rpe_error):
    #static和error 三列：[ape,rpe_t,rpe_r]
    np.savetxt(os.path.join(ppath,"static"+mode+".txt"),static.T)
    np.savetxt(os.path.join(ppath,"ape_error"+mode+".txt"),ape_error.T)
    np.savetxt(os.path.join(ppath,"rpe_error"+mode+".txt"),rpe_error.T)

def main():
    results_path=r'../results/'#input
    output_path=r'../errors/'#output
    algs=['A-LOAM','LeGO-LOAM_f','SC-LeGO-LOAM']#algorithms
    for i in range(0,11): #seq
        print(i)
        c_index=str(i).zfill(2) #seq format
        data_path=os.path.join(results_path,c_index)#
        ref_poses=os.path.join(data_path,c_index+'.txt')# 以序号命名的为kitti，SUMA提供的为poses.txt文件
        ref_times=os.path.join(data_path,'times.txt')
        # assert ref pose and time
        assert os.path.exists(ref_poses) and os.path.exists(ref_times)
        for a in algs:
            est_poses_full=os.path.join(data_path,a,c_index+'_raw.txt')
            est_poses_static=os.path.join(data_path,a,c_index+'_d.txt')
            #print(est_poses_full,est_poses_static)
            # assert est pose
            assert os.path.exists(est_poses_full) and os.path.exists(est_poses_static)

            outpath=os.path.join(output_path,c_index,a)
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            if a == 'A-LOAM':
                #共享原始数据时间戳=ref_times

                # 评估完整数据SLAM结果
                static_raw,ape_error_raw,rpe_error_raw=eval(ref_poses,ref_times,est_poses_full,ref_times)
                save_results(outpath,'_raw',static_raw,ape_error_raw,rpe_error_raw)
                # 评估过滤数据SLAM结果
                static_d,ape_error_d,rpe_error_d=eval(ref_poses,ref_times,est_poses_static,ref_times)
                save_results(outpath,'_d',static_d,ape_error_d,rpe_error_d)
            else:

                #使用算法估算时间戳，
                est_time_full=os.path.join(data_path,a,'time_raw.txt')
                est_time_static=os.path.join(data_path,a,'time_d.txt')
                #assert est time
                assert os.path.exists(est_time_full) and os.path.exists(est_time_static)
                # 评估完整数据SLAM结果，使用est_time_full
                static_raw,ape_error_raw,rpe_error_raw=eval(ref_poses,ref_times,est_poses_full,est_time_full)
                save_results(outpath,'_raw',static_raw,ape_error_raw,rpe_error_raw)
                # 评估过滤数据SLAM结果，使用est_time_static
                static_d,ape_error_d,rpe_error_d=eval(ref_poses,ref_times,est_poses_static,est_time_static)
                save_results(outpath,'_d',static_d,ape_error_d,rpe_error_d)

if __name__ == "__main__":
    main()