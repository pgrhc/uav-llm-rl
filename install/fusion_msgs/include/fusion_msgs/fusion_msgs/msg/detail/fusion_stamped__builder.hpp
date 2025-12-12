// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__BUILDER_HPP_
#define FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fusion_msgs/msg/detail/fusion_stamped__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fusion_msgs
{

namespace msg
{

namespace builder
{

class Init_FusionStamped_odom
{
public:
  explicit Init_FusionStamped_odom(::fusion_msgs::msg::FusionStamped & msg)
  : msg_(msg)
  {}
  ::fusion_msgs::msg::FusionStamped odom(::fusion_msgs::msg::FusionStamped::_odom_type arg)
  {
    msg_.odom = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fusion_msgs::msg::FusionStamped msg_;
};

class Init_FusionStamped_radar
{
public:
  explicit Init_FusionStamped_radar(::fusion_msgs::msg::FusionStamped & msg)
  : msg_(msg)
  {}
  Init_FusionStamped_odom radar(::fusion_msgs::msg::FusionStamped::_radar_type arg)
  {
    msg_.radar = std::move(arg);
    return Init_FusionStamped_odom(msg_);
  }

private:
  ::fusion_msgs::msg::FusionStamped msg_;
};

class Init_FusionStamped_lidar
{
public:
  explicit Init_FusionStamped_lidar(::fusion_msgs::msg::FusionStamped & msg)
  : msg_(msg)
  {}
  Init_FusionStamped_radar lidar(::fusion_msgs::msg::FusionStamped::_lidar_type arg)
  {
    msg_.lidar = std::move(arg);
    return Init_FusionStamped_radar(msg_);
  }

private:
  ::fusion_msgs::msg::FusionStamped msg_;
};

class Init_FusionStamped_image
{
public:
  explicit Init_FusionStamped_image(::fusion_msgs::msg::FusionStamped & msg)
  : msg_(msg)
  {}
  Init_FusionStamped_lidar image(::fusion_msgs::msg::FusionStamped::_image_type arg)
  {
    msg_.image = std::move(arg);
    return Init_FusionStamped_lidar(msg_);
  }

private:
  ::fusion_msgs::msg::FusionStamped msg_;
};

class Init_FusionStamped_stamp
{
public:
  Init_FusionStamped_stamp()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FusionStamped_image stamp(::fusion_msgs::msg::FusionStamped::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return Init_FusionStamped_image(msg_);
  }

private:
  ::fusion_msgs::msg::FusionStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fusion_msgs::msg::FusionStamped>()
{
  return fusion_msgs::msg::builder::Init_FusionStamped_stamp();
}

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__BUILDER_HPP_
