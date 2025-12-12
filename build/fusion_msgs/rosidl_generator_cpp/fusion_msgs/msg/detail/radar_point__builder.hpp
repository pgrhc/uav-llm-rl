// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from fusion_msgs:msg/RadarPoint.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINT__BUILDER_HPP_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "fusion_msgs/msg/detail/radar_point__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace fusion_msgs
{

namespace msg
{

namespace builder
{

class Init_RadarPoint_elevation
{
public:
  explicit Init_RadarPoint_elevation(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  ::fusion_msgs::msg::RadarPoint elevation(::fusion_msgs::msg::RadarPoint::_elevation_type arg)
  {
    msg_.elevation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_azimuth
{
public:
  explicit Init_RadarPoint_azimuth(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  Init_RadarPoint_elevation azimuth(::fusion_msgs::msg::RadarPoint::_azimuth_type arg)
  {
    msg_.azimuth = std::move(arg);
    return Init_RadarPoint_elevation(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_range
{
public:
  explicit Init_RadarPoint_range(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  Init_RadarPoint_azimuth range(::fusion_msgs::msg::RadarPoint::_range_type arg)
  {
    msg_.range = std::move(arg);
    return Init_RadarPoint_azimuth(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_intensity
{
public:
  explicit Init_RadarPoint_intensity(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  Init_RadarPoint_range intensity(::fusion_msgs::msg::RadarPoint::_intensity_type arg)
  {
    msg_.intensity = std::move(arg);
    return Init_RadarPoint_range(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_z
{
public:
  explicit Init_RadarPoint_z(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  Init_RadarPoint_intensity z(::fusion_msgs::msg::RadarPoint::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_RadarPoint_intensity(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_y
{
public:
  explicit Init_RadarPoint_y(::fusion_msgs::msg::RadarPoint & msg)
  : msg_(msg)
  {}
  Init_RadarPoint_z y(::fusion_msgs::msg::RadarPoint::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_RadarPoint_z(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

class Init_RadarPoint_x
{
public:
  Init_RadarPoint_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_RadarPoint_y x(::fusion_msgs::msg::RadarPoint::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_RadarPoint_y(msg_);
  }

private:
  ::fusion_msgs::msg::RadarPoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::fusion_msgs::msg::RadarPoint>()
{
  return fusion_msgs::msg::builder::Init_RadarPoint_x();
}

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINT__BUILDER_HPP_
