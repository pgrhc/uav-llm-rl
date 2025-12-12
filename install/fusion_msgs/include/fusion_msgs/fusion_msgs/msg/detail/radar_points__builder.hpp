// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__BUILDER_HPP_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fusion_msgs/msg/detail/radar_points__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fusion_msgs
{

namespace msg
{

namespace builder
{

class Init_RadarPoints_points
{
public:
  explicit Init_RadarPoints_points(::fusion_msgs::msg::RadarPoints & msg)
  : msg_(msg)
  {}
  ::fusion_msgs::msg::RadarPoints points(::fusion_msgs::msg::RadarPoints::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoints msg_;
};

class Init_RadarPoints_header
{
public:
  Init_RadarPoints_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RadarPoints_points header(::fusion_msgs::msg::RadarPoints::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_RadarPoints_points(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoints msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fusion_msgs::msg::RadarPoints>()
{
  return fusion_msgs::msg::builder::Init_RadarPoints_header();
}

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__BUILDER_HPP_
