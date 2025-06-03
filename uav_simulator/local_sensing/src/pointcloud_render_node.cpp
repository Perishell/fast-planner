#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <pcl/search/impl/kdtree.hpp>
#include <vector>

using namespace std;
using namespace Eigen;

ros::Publisher pub_cloud;

sensor_msgs::PointCloud2 local_map_pcl;
sensor_msgs::PointCloud2 local_depth_pcl;

ros::Subscriber odom_sub;
ros::Subscriber global_map_sub, local_map_sub;

ros::Timer local_sensing_timer;

bool has_global_map(false);
bool has_local_map(false);
bool has_odom(false);

nav_msgs::Odometry _odom;

double sensing_horizon, sensing_rate, estimation_rate;
double _x_size, _y_size, _z_size;
double _gl_xl, _gl_yl, _gl_zl;
double _resolution, _inv_resolution;
int _GLX_SIZE, _GLY_SIZE, _GLZ_SIZE;

ros::Time last_odom_stamp = ros::TIME_MAX;

inline Eigen::Vector3d gridIndex2coord(const Eigen::Vector3i& index) {
  Eigen::Vector3d pt;
  pt(0) = ((double)index(0) + 0.5) * _resolution + _gl_xl;
  pt(1) = ((double)index(1) + 0.5) * _resolution + _gl_yl;
  pt(2) = ((double)index(2) + 0.5) * _resolution + _gl_zl;

  return pt;
};

inline Eigen::Vector3i coord2gridIndex(const Eigen::Vector3d& pt) {
  Eigen::Vector3i idx;
  idx(0) = std::min(std::max(int((pt(0) - _gl_xl) * _inv_resolution), 0),
                    _GLX_SIZE - 1);
  idx(1) = std::min(std::max(int((pt(1) - _gl_yl) * _inv_resolution), 0),
                    _GLY_SIZE - 1);
  idx(2) = std::min(std::max(int((pt(2) - _gl_zl) * _inv_resolution), 0),
                    _GLZ_SIZE - 1);

  return idx;
};

void rcvOdometryCallbck(const nav_msgs::Odometry& odom) {
  /*if(!has_global_map)
    return;*/
  has_odom = true;
  _odom = odom;
}

pcl::PointCloud<pcl::PointXYZ> _cloud_all_map, _local_map;
pcl::VoxelGrid<pcl::PointXYZ> _voxel_sampler;
sensor_msgs::PointCloud2 _local_map_pcd;

pcl::search::KdTree<pcl::PointXYZ> _kdtreeLocalMap;
vector<int> _pointIdxRadiusSearch;
vector<float> _pointRadiusSquaredDistance;

void rcvGlobalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  if (has_global_map) return;

  ROS_WARN("Global Pointcloud received..");

  pcl::PointCloud<pcl::PointXYZ> cloud_input;
  pcl::fromROSMsg(pointcloud_map, cloud_input);

  _voxel_sampler.setLeafSize(0.1f, 0.1f, 0.1f);
  _voxel_sampler.setInputCloud(cloud_input.makeShared());
  _voxel_sampler.filter(_cloud_all_map);

  _kdtreeLocalMap.setInputCloud(_cloud_all_map.makeShared());

  has_global_map = true;
}

void renderSensedPoints(const ros::TimerEvent& event) {
  // 检查全局地图和里程计数据是否就绪
  if (!has_global_map || !has_odom) return;

  // 从里程计获取当前姿态四元数
  Eigen::Quaterniond q;
  q.x() = _odom.pose.pose.orientation.x; // X分量
  q.y() = _odom.pose.pose.orientation.y; // Y分量
  q.z() = _odom.pose.pose.orientation.z; // Z分量
  q.w() = _odom.pose.pose.orientation.w; // 实部

  // 将四元数转换为旋转矩阵
  Eigen::Matrix3d rot;
  rot = q;
  // 获取Yaw轴（通常为机器人的前进方向）
  Eigen::Vector3d yaw_vec = rot.col(0);

  // 清空局部地图点云缓存
  _local_map.points.clear();

  // 创建以当前位置为中心的搜索点
  pcl::PointXYZ searchPoint(_odom.pose.pose.position.x,
                            _odom.pose.pose.position.y,
                            _odom.pose.pose.position.z);
  // 清空KD树搜索结果缓存    
  _pointIdxRadiusSearch.clear();
  _pointRadiusSquaredDistance.clear();

  // 执行半径搜索（核心感知逻辑）
  pcl::PointXYZ pt;
  if (_kdtreeLocalMap.radiusSearch(
    // 使用KD树加速搜索
    searchPoint,                      // 搜索中心点
    sensing_horizon,                  // 感知半径（单位：米）
    _pointIdxRadiusSearch,            // 输出：找到的点索引
    _pointRadiusSquaredDistance) > 0) // 输出：平方距离
    {
    // 遍历所有找到的点
    for (size_t i = 0; i < _pointIdxRadiusSearch.size(); ++i) {
      // 从全局地图获取点数据
      pt = _cloud_all_map.points[_pointIdxRadiusSearch[i]];
      /* ------ 高度过滤：排除垂直方向视野外的点 ------ */
      // 计算高度差与水平距离的比值（正切值）
      // 若超过15度（tan(π/12)≈0.2679）则跳过
      if ((fabs(pt.z - _odom.pose.pose.position.z) / (sensing_horizon)) >
          tan(M_PI / 12.0))
        continue;
      /* ------ 前方过滤：排除机器人后方的点 ------ */
      // 计算点相对于机器人的位置向量
      Vector3d pt_vec(pt.x - _odom.pose.pose.position.x,
                      pt.y - _odom.pose.pose.position.y,
                      pt.z - _odom.pose.pose.position.z);
      // 通过过滤的点加入局部地图
      if (pt_vec.dot(yaw_vec) < 0) continue;

      _local_map.points.push_back(pt);
    }
  } else {
    return;
  }

  _local_map.width = _local_map.points.size();
  _local_map.height = 1;
  _local_map.is_dense = true;

  pcl::toROSMsg(_local_map, _local_map_pcd);
  _local_map_pcd.header.frame_id = "map";

  pub_cloud.publish(_local_map_pcd);
}

void rcvLocalPointCloudCallBack(
    const sensor_msgs::PointCloud2& pointcloud_map) {
  // do nothing, fix later
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_render");
  ros::NodeHandle nh("~");

  nh.getParam("sensing_horizon", sensing_horizon);
  nh.getParam("sensing_rate", sensing_rate);
  nh.getParam("estimation_rate", estimation_rate);

  nh.getParam("map/x_size", _x_size);
  nh.getParam("map/y_size", _y_size);
  nh.getParam("map/z_size", _z_size);

  // subscribe point cloud
  // 订阅的是从random_forest_sensing.cpp中来的/map_generator/global_cloud话题。
  global_map_sub = nh.subscribe("global_map", 1, rcvGlobalPointCloudCallBack);
  // 这个订阅部分啥也没有，回调函数里面为空
  local_map_sub = nh.subscribe("local_map", 1, rcvLocalPointCloudCallBack);
  // 订阅odom
  odom_sub = nh.subscribe("odometry", 50, rcvOdometryCallbck);
  // publisher depth image and color image
  pub_cloud =
      nh.advertise<sensor_msgs::PointCloud2>("/pcl_render_node/cloud", 10);

  // 将原始时间间隔乘以一个系数，生成新的触发间隔 控制频率
  double sensing_duration = 1.0 / sensing_rate * 2.5;

  // 创建一个ROS定时器，周期性执行指定回调函数
  local_sensing_timer =
      nh.createTimer(ros::Duration(sensing_duration), renderSensedPoints);

  _inv_resolution = 1.0 / _resolution;

  _gl_xl = -_x_size / 2.0;
  _gl_yl = -_y_size / 2.0;
  _gl_zl = 0.0;

  _GLX_SIZE = (int)(_x_size * _inv_resolution);
  _GLY_SIZE = (int)(_y_size * _inv_resolution);
  _GLZ_SIZE = (int)(_z_size * _inv_resolution);

  ros::Rate rate(100);
  bool status = ros::ok();
  while (status) {
    ros::spinOnce();
    status = ros::ok();
    rate.sleep();
  }
}
