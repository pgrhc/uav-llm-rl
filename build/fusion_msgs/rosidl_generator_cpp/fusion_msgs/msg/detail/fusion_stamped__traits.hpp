// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__TRAITS_HPP_
#define FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fusion_msgs/msg/detail/fusion_stamped__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"
// Member 'image'
#include "sensor_msgs/msg/detail/image__traits.hpp"
// Member 'lidar'
#include "sensor_msgs/msg/detail/point_cloud2__traits.hpp"
// Member 'radar'
#include "fusion_msgs/msg/detail/radar_points__traits.hpp"
// Member 'odom'
#include "nav_msgs/msg/detail/odometry__traits.hpp"

namespace fusion_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const FusionStamped & msg,
  std::ostream & out)
{
  out << "{";
  // member: stamp
  {
    out << "stamp: ";
    to_flow_style_yaml(msg.stamp, out);
    out << ", ";
  }

  // member: image
  {
    out << "image: ";
    to_flow_style_yaml(msg.image, out);
    out << ", ";
  }

  // member: lidar
  {
    out << "lidar: ";
    to_flow_style_yaml(msg.lidar, out);
    out << ", ";
  }

  // member: radar
  {
    out << "radar: ";
    to_flow_style_yaml(msg.radar, out);
    out << ", ";
  }

  // member: odom
  {
    out << "odom: ";
    to_flow_style_yaml(msg.odom, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const FusionStamped & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_block_style_yaml(msg.stamp, out, indentation + 2);
  }

  // member: image
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image:\n";
    to_block_style_yaml(msg.image, out, indentation + 2);
  }

  // member: lidar
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lidar:\n";
    to_block_style_yaml(msg.lidar, out, indentation + 2);
  }

  // member: radar
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "radar:\n";
    to_block_style_yaml(msg.radar, out, indentation + 2);
  }

  // member: odom
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "odom:\n";
    to_block_style_yaml(msg.odom, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const FusionStamped & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace fusion_msgs

namespace rosidl_generator_traits
{

[[deprecated("use fusion_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const fusion_msgs::msg::FusionStamped & msg,
  std::ostream & out, size_t indentation = 0)
{
  fusion_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fusion_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const fusion_msgs::msg::FusionStamped & msg)
{
  return fusion_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fusion_msgs::msg::FusionStamped>()
{
  return "fusion_msgs::msg::FusionStamped";
}

template<>
inline const char * name<fusion_msgs::msg::FusionStamped>()
{
  return "fusion_msgs/msg/FusionStamped";
}

template<>
struct has_fixed_size<fusion_msgs::msg::FusionStamped>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value && has_fixed_size<fusion_msgs::msg::RadarPoints>::value && has_fixed_size<nav_msgs::msg::Odometry>::value && has_fixed_size<sensor_msgs::msg::Image>::value && has_fixed_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct has_bounded_size<fusion_msgs::msg::FusionStamped>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value && has_bounded_size<fusion_msgs::msg::RadarPoints>::value && has_bounded_size<nav_msgs::msg::Odometry>::value && has_bounded_size<sensor_msgs::msg::Image>::value && has_bounded_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct is_message<fusion_msgs::msg::FusionStamped>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__TRAITS_HPP_
