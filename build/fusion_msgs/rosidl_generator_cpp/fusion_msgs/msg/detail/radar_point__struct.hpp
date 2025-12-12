// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fusion_msgs:msg/RadarPoint.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_HPP_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__fusion_msgs__msg__RadarPoint __attribute__((deprecated))
#else
# define DEPRECATED__fusion_msgs__msg__RadarPoint __declspec(deprecated)
#endif

namespace fusion_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RadarPoint_
{
  using Type = RadarPoint_<ContainerAllocator>;

  explicit RadarPoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0f;
      this->y = 0.0f;
      this->z = 0.0f;
      this->intensity = 0.0f;
      this->range = 0.0f;
      this->azimuth = 0.0f;
      this->elevation = 0.0f;
    }
  }

  explicit RadarPoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0f;
      this->y = 0.0f;
      this->z = 0.0f;
      this->intensity = 0.0f;
      this->range = 0.0f;
      this->azimuth = 0.0f;
      this->elevation = 0.0f;
    }
  }

  // field types and members
  using _x_type =
    float;
  _x_type x;
  using _y_type =
    float;
  _y_type y;
  using _z_type =
    float;
  _z_type z;
  using _intensity_type =
    float;
  _intensity_type intensity;
  using _range_type =
    float;
  _range_type range;
  using _azimuth_type =
    float;
  _azimuth_type azimuth;
  using _elevation_type =
    float;
  _elevation_type elevation;

  // setters for named parameter idiom
  Type & set__x(
    const float & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const float & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const float & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__intensity(
    const float & _arg)
  {
    this->intensity = _arg;
    return *this;
  }
  Type & set__range(
    const float & _arg)
  {
    this->range = _arg;
    return *this;
  }
  Type & set__azimuth(
    const float & _arg)
  {
    this->azimuth = _arg;
    return *this;
  }
  Type & set__elevation(
    const float & _arg)
  {
    this->elevation = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fusion_msgs::msg::RadarPoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const fusion_msgs::msg::RadarPoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::RadarPoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::RadarPoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fusion_msgs__msg__RadarPoint
    std::shared_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fusion_msgs__msg__RadarPoint
    std::shared_ptr<fusion_msgs::msg::RadarPoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RadarPoint_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->intensity != other.intensity) {
      return false;
    }
    if (this->range != other.range) {
      return false;
    }
    if (this->azimuth != other.azimuth) {
      return false;
    }
    if (this->elevation != other.elevation) {
      return false;
    }
    return true;
  }
  bool operator!=(const RadarPoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RadarPoint_

// alias to use template instance with default allocator
using RadarPoint =
  fusion_msgs::msg::RadarPoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINT__STRUCT_HPP_
