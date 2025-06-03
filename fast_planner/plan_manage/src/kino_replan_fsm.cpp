/**
* This file is part of Fast-Planner.
*
* Copyright 2019 Boyu Zhou, Aerial Robotics Group, Hong Kong University of Science and Technology, <uav.ust.hk>
* Developed by Boyu Zhou <bzhouai at connect dot ust dot hk>, <uv dot boyuzhou at gmail dot com>
* for more information see <https://github.com/HKUST-Aerial-Robotics/Fast-Planner>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* Fast-Planner is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Fast-Planner is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with Fast-Planner. If not, see <http://www.gnu.org/licenses/>.
*/




#include <plan_manage/kino_replan_fsm.h>

namespace fast_planner {

// 初始化有限状态机（FSM）
void KinoReplanFSM::init(ros::NodeHandle& nh) {
  // 初始化状态变量
  current_wp_  = 0;                          // 当前航点索引（从0开始）
  exec_state_  = FSM_EXEC_STATE::INIT;       // 初始状态：初始化阶段
  have_target_ = false;                      // 目标点未接收标志
  have_odom_   = false;                      // 里程计未接收标志

  /*  fsm param  */
  // 飞行类型（用rviz指引导航点/写入到launch文件中的导航点）
  nh.param("fsm/flight_type", target_type_, -1);
  // 路径重规划阈值（当偏离路径超过此值时触发重规划）                          // 这两个参数的launch有一些自相矛盾？？？？？
  nh.param("fsm/thresh_replan", replan_thresh_, -1.0);
  // 免重规划阈值（当路径误差小于此值时保持当前路径）
  nh.param("fsm/thresh_no_replan", no_replan_thresh_, -1.0);

  // 加载航点数量（例如：3个导航点）
  nh.param("fsm/waypoint_num", waypoint_num_, -1);

  // 遍历加载每个航点的三维坐标
  for (int i = 0; i < waypoint_num_; i++) {
    nh.param("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0], -1.0);
    nh.param("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1], -1.0);
    nh.param("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2], -1.0);
  }

  /*---------------------- 初始化核心功能模块 ----------------------*/
  planner_manager_.reset(new FastPlannerManager);      // 创建运动规划管理器
  planner_manager_->initPlanModules(nh);               // 初始化规划模块（包含地图、优化器等）
  visualization_.reset(new PlanningVisualization(nh)); // 创建可视化工具

  /* callback */ // 相关kinodynamic A*代码在execFSMCallback中出现
  exec_timer_   = nh.createTimer(ros::Duration(0.01), &KinoReplanFSM::execFSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &KinoReplanFSM::checkCollisionCallback, this);

  waypoint_sub_ =
      nh.subscribe("/waypoint_generator/waypoints", 1, &KinoReplanFSM::waypointCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &KinoReplanFSM::odometryCallback, this);

  replan_pub_  = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_     = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<plan_manage::Bspline>("/planning/bspline", 10);
}

void KinoReplanFSM::waypointCallback(const nav_msgs::PathConstPtr& msg) {
  // 过滤无效目标点，若接收到的第一个路径点的z坐标低于-0.1（可能表示地面或无效高度），直接返回不处理。
  if (msg->poses[0].pose.position.z < -0.1) return;
  // 打印触发日志，并设置trigger_标志为true，表示已接收到有效目标点
  cout << "Triggered!" << endl;
  trigger_ = true;

  if (target_type_ == TARGET_TYPE::MANUAL_TARGET) {
    // 从消息中提取目标点的x、y坐标，z固定为1.0（默认飞行高度）
    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 1.0;
    // 从预定义的航点列表waypoints_中按顺序循环读取目标点坐标
  } else if (target_type_ == TARGET_TYPE::PRESET_TARGET) {
    end_pt_(0)  = waypoints_[current_wp_][0];
    end_pt_(1)  = waypoints_[current_wp_][1];
    end_pt_(2)  = waypoints_[current_wp_][2];
    current_wp_ = (current_wp_ + 1) % waypoint_num_;
  }

  visualization_->drawGoal(end_pt_, 0.3, Eigen::Vector4d(1, 0, 0, 1.0));
  end_vel_.setZero();
  have_target_ = true;

  if (exec_state_ == WAIT_TARGET)
    // 若当前状态为等待目标，则生成新轨迹
    changeFSMExecState(GEN_NEW_TRAJ, "TRIG");

  else if (exec_state_ == EXEC_TRAJ)
    // 若正在执行轨迹，则触发重规划
    changeFSMExecState(REPLAN_TRAJ, "TRIG");
}

void KinoReplanFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  odom_pos_(0) = msg->pose.pose.position.x;
  odom_pos_(1) = msg->pose.pose.position.y;
  odom_pos_(2) = msg->pose.pose.position.z;

  odom_vel_(0) = msg->twist.twist.linear.x;
  odom_vel_(1) = msg->twist.twist.linear.y;
  odom_vel_(2) = msg->twist.twist.linear.z;

  odom_orient_.w() = msg->pose.pose.orientation.w;
  odom_orient_.x() = msg->pose.pose.orientation.x;
  odom_orient_.y() = msg->pose.pose.orientation.y;
  odom_orient_.z() = msg->pose.pose.orientation.z;

  have_odom_ = true;
}

void KinoReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call) {
  string state_str[5] = { "INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ" };
  int    pre_s        = int(exec_state_);
  exec_state_         = new_state;
  cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
}

void KinoReplanFSM::printFSMExecState() {
  string state_str[5] = { "INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ" };

  cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
}

void KinoReplanFSM::execFSMCallback(const ros::TimerEvent& e) {
  // 状态机执行计数器，用于定期打印状态信息
  static int fsm_num = 0;
  fsm_num++;
  // 每执行100次（约1秒，假设定时器周期为10ms）打印一次状态
  if (fsm_num == 100) {
    printFSMExecState();
    if (!have_odom_) cout << "no odom." << endl;
    if (!trigger_) cout << "wait for goal." << endl;
    fsm_num = 0;
  }
  // 状态机核心逻辑
  switch (exec_state_) {
    // 初始化状态
    case INIT: {
      // 等待必要条件：收到里程计和触发信号
      if (!have_odom_) {
        return;
      }
      if (!trigger_) {
        return;
      }
      // 满足条件后进入等待目标状态
      changeFSMExecState(WAIT_TARGET, "FSM");
      break;
    }
    // 等待目标状态
    case WAIT_TARGET: {
      if (!have_target_)
        return;
      else {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }
    // 生成新轨迹状态
    case GEN_NEW_TRAJ: {
      // 设置轨迹起点状态：位置、速度、加速度
      start_pt_  = odom_pos_; // 从里程计获取当前位置
      start_vel_ = odom_vel_; // 从里程计获取当前速度
      start_acc_.setZero();  // 假设初始加速度为0

      // 计算当前偏航角(yaw)
      Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
      start_yaw_(0)         = atan2(rot_x(1), rot_x(0)); // 从旋转矩阵提取偏航角
      start_yaw_(1) = start_yaw_(2) = 0.0;              // 角速度和角加速度初始化为0

      // 调用运动学规划器生成轨迹
      bool success = callKinodynamicReplan();
      if (success) {
        // 成功生成则执行轨迹
        changeFSMExecState(EXEC_TRAJ, "FSM");
      } else {
        // have_target_ = false;
        // changeFSMExecState(WAIT_TARGET, "FSM");
        // 若失败则保持当前状态继续尝试生成
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }
    // 执行轨迹状态
    case EXEC_TRAJ: {
      /* determine if need to replan */
      // 获取当前轨迹信息
      LocalTrajData* info     = &planner_manager_->local_data_;
      ros::Time      time_now = ros::Time::now();
      double         t_cur    = (time_now - info->start_time_).toSec();
      t_cur                   = min(info->duration_, t_cur);

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t_cur);

      /* && (end_pt_ - pos).norm() < 0.5 */
      if (t_cur > info->duration_ - 1e-2) {
        have_target_ = false;
        changeFSMExecState(WAIT_TARGET, "FSM");
        return;

      } else if ((end_pt_ - pos).norm() < no_replan_thresh_) {
        // cout << "near end" << endl;
        return;

      } else if ((info->start_pos_ - pos).norm() < replan_thresh_) {
        // cout << "near start" << endl;
        return;

      } else {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }
      break;
    }

    case REPLAN_TRAJ: {
      LocalTrajData* info     = &planner_manager_->local_data_;
      ros::Time      time_now = ros::Time::now();
      double         t_cur    = (time_now - info->start_time_).toSec();

      start_pt_  = info->position_traj_.evaluateDeBoorT(t_cur);
      start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
      start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);

      start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_cur)[0];
      start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_cur)[0];
      start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_cur)[0];

      std_msgs::Empty replan_msg;
      replan_pub_.publish(replan_msg);

      bool success = callKinodynamicReplan();
      if (success) {
        changeFSMExecState(EXEC_TRAJ, "FSM");
      } else {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }
  }
}

void KinoReplanFSM::checkCollisionCallback(const ros::TimerEvent& e) {
  LocalTrajData* info = &planner_manager_->local_data_;

  if (have_target_) {
    auto edt_env = planner_manager_->edt_environment_;

    double dist = planner_manager_->pp_.dynamic_ ?
        edt_env->evaluateCoarseEDT(end_pt_, /* time to program start + */ info->duration_) :
        edt_env->evaluateCoarseEDT(end_pt_, -1.0);

    if (dist <= 0.3) {
      /* try to find a max distance goal around */
      bool            new_goal = false;
      const double    dr = 0.5, dtheta = 30, dz = 0.3;
      double          new_x, new_y, new_z, max_dist = -1.0;
      Eigen::Vector3d goal;

      for (double r = dr; r <= 5 * dr + 1e-3; r += dr) {
        for (double theta = -90; theta <= 270; theta += dtheta) {
          for (double nz = 1 * dz; nz >= -1 * dz; nz -= dz) {

            new_x = end_pt_(0) + r * cos(theta / 57.3);
            new_y = end_pt_(1) + r * sin(theta / 57.3);
            new_z = end_pt_(2) + nz;

            Eigen::Vector3d new_pt(new_x, new_y, new_z);
            dist = planner_manager_->pp_.dynamic_ ?
                edt_env->evaluateCoarseEDT(new_pt, /* time to program start+ */ info->duration_) :
                edt_env->evaluateCoarseEDT(new_pt, -1.0);

            if (dist > max_dist) {
              /* reset end_pt_ */
              goal(0)  = new_x;
              goal(1)  = new_y;
              goal(2)  = new_z;
              max_dist = dist;
            }
          }
        }
      }

      if (max_dist > 0.3) {
        cout << "change goal, replan." << endl;
        end_pt_      = goal;
        have_target_ = true;
        end_vel_.setZero();

        if (exec_state_ == EXEC_TRAJ) {
          changeFSMExecState(REPLAN_TRAJ, "SAFETY");
        }

        visualization_->drawGoal(end_pt_, 0.3, Eigen::Vector4d(1, 0, 0, 1.0));
      } else {
        // have_target_ = false;
        // cout << "Goal near collision, stop." << endl;
        // changeFSMExecState(WAIT_TARGET, "SAFETY");
        cout << "goal near collision, keep retry" << endl;
        changeFSMExecState(REPLAN_TRAJ, "FSM");

        std_msgs::Empty emt;
        replan_pub_.publish(emt);
      }
    }
  }

  /* ---------- check trajectory ---------- */
  if (exec_state_ == FSM_EXEC_STATE::EXEC_TRAJ) {
    double dist;
    bool   safe = planner_manager_->checkTrajCollision(dist);

    if (!safe) {
      // cout << "current traj in collision." << endl;
      ROS_WARN("current traj in collision.");
      changeFSMExecState(REPLAN_TRAJ, "SAFETY");
    }
  }
}

bool KinoReplanFSM::callKinodynamicReplan() {

  // 调用规划器的运动学动态重规划接口（核心规划逻辑）
  // 参数包含：起点位置、起点速度、起点加速度、终点位置、终点速度
  bool plan_success =
      planner_manager_->kinodynamicReplan(start_pt_, start_vel_, start_acc_, end_pt_, end_vel_);

  if (plan_success) {

    planner_manager_->planYaw(start_yaw_);

    auto info = &planner_manager_->local_data_;

    /* publish traj */
    plan_manage::Bspline bspline;
    bspline.order      = 3;
    bspline.start_time = info->start_time_;
    bspline.traj_id    = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();

    for (int i = 0; i < pos_pts.rows(); ++i) {
      geometry_msgs::Point pt;
      pt.x = pos_pts(i, 0);
      pt.y = pos_pts(i, 1);
      pt.z = pos_pts(i, 2);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    for (int i = 0; i < knots.rows(); ++i) {
      bspline.knots.push_back(knots(i));
    }

    Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
    for (int i = 0; i < yaw_pts.rows(); ++i) {
      double yaw = yaw_pts(i, 0);
      bspline.yaw_pts.push_back(yaw);
    }
    bspline.yaw_dt = info->yaw_traj_.getInterval();

    bspline_pub_.publish(bspline);

    /* visulization */
    auto plan_data = &planner_manager_->plan_data_;
    visualization_->drawGeometricPath(plan_data->kino_path_, 0.075, Eigen::Vector4d(1, 1, 0, 0.4));
    visualization_->drawBspline(info->position_traj_, 0.1, Eigen::Vector4d(1.0, 0, 0.0, 1), true, 0.2,
                                Eigen::Vector4d(1, 0, 0, 1));

    return true;

  } else {
    cout << "generate new traj fail." << endl;
    return false;
  }
}

// KinoReplanFSM::
}  // namespace fast_planner
