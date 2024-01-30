
import numpy as np
import open3d as o3d

def pcd_show(point_clouds=[]):
    show_list = [];
    for point_cloud in point_clouds:
        if(type(point_cloud).__module__ == np.__name__):
            np_point_cloud = np.array(point_cloud);
            np_point_cloud = np_point_cloud.reshape((-1,3));
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(np_point_cloud));
            show_list.append(o3d_point_cloud);
        else:
            show_list.append(point_cloud);
    o3d.visualization.draw_geometries( show_list,point_show_normal=False);

def pcd_rotation(point_cloud,roll_deg=0.0,pitch_deg=0.0,yaw_deg=0.0):
    roll_T = np.array([[1,0,0],
                       [0,np.cos(np.deg2rad(roll_deg)),-np.sin(np.deg2rad(roll_deg))],
                       [0,np.sin(np.deg2rad(roll_deg)),np.cos(np.deg2rad(roll_deg))],
                       ])
    pitch_T = np.array([[np.cos(np.deg2rad(pitch_deg)),0,np.sin(np.deg2rad(pitch_deg))],
                       [0,1,0],
                       [-np.sin(np.deg2rad(pitch_deg)),0,np.cos(np.deg2rad(pitch_deg))],
                       ])
    yaw_T = np.array([[np.cos(np.deg2rad(yaw_deg)),-np.sin(np.deg2rad(yaw_deg)),0],
                       [np.sin(np.deg2rad(yaw_deg)),np.cos(np.deg2rad(yaw_deg)),0],
                       [0,0,1],
                       ])
    np_point_cloud = point_cloud.reshape((-1,3));
    t_pcd = np.matmul(np_point_cloud,np.matmul(np.matmul(yaw_T,pitch_T),roll_T));
    return t_pcd;

if __name__ == "__main__":
    coord=o3d.geometry.TriangleMesh.create_coordinate_frame();
    theta_sample_num = 100;
    theta = np.arange(0.0,2*np.pi,(2*np.pi/100));
    r = 1.0
    x = r*np.cos(theta);
    y = r*np.sin(theta);
    z = np.zeros_like(x);

    target=np.stack([x,y,z],axis=-1);
    source=pcd_rotation(target,45.0,0,0);

    pcd_show([source,target])