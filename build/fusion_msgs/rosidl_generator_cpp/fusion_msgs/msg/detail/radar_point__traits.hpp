// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from fusion_msgs:msg/RadarPoint.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINT__TRAITS_HPP_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "fusion_msgs/msg/detail/radar_point__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace fusion_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const RadarPoint & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: intensity
  {
    out << "intensity: ";
    rosidl_generator_traits::value_to_yaml(msg.intensity, out);
    out << ", ";
  }

  // member: range
  {
    out << "range: ";
    rosidl_generator_traits::value_to_yaml(msg.range, out);
    out << ", ";
  }

  // member: azimuth
  {
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << ", ";
  }

  // member: elevation
  {
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const RadarPoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: intensity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "intensity: ";
    rosidl_generator_traits::value_to_yaml(msg.intensity, out);
    out << "\n";
  }

  // member: range
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "range: ";
    rosidl_generator_traits::value_to_yaml(msg.range, out);
    out << "\n";
  }

  // member: azimuth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "azimuth: ";
    rosidl_generator_traits::value_to_yaml(msg.azimuth, out);
    out << "\n";
  }

  // member: elevation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "elevation: ";
    rosidl_generator_traits::value_to_yaml(msg.elevation, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const RadarPoint & msg, bool use_flow_style = false)
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
  const fusion_msgs::msg::RadarPoint & msg,
  std::ostream & out, size_t indentation = 0)
{
  fusion_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use fusion_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const fusion_msgs::msg::RadarPoint & msg)
{
  return fusion_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<fusion_msgs::msg::RadarPoint>()
{
  return "fusion_msgs::msg::RadarPoint";
}

template<>
inline const char * name<fusion_msgs::msg::RadarPoint>()
{
  return "fusion_msgs/msg/RadarPoint";
}

template<>
struct has_fixed_size<fusion_msgs::msg::RadarPoint>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<fusion_msgs::msg::RadarPoint>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<fusion_msgs::msg::RadarPoint>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINT__TRAITS_HPP_
