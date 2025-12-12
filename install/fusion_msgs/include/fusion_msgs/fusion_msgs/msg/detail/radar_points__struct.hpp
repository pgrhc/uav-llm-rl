// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fusion_msgs:msg/RadarPoints.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_HPP_
#define FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'points'
#include "fusion_msgs/msg/detail/radar_point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__fusion_msgs__msg__RadarPoints __attribute__((deprecated))
#else
# define DEPRECATED__fusion_msgs__msg__RadarPoints __declspec(deprecated)
#endif

namespace fusion_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct RadarPoints_
{
  using Type = RadarPoints_<ContainerAllocator>;

  explicit RadarPoints_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit RadarPoints_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _points_type =
    std::vector<fusion_msgs::msg::RadarPoint_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fusion_msgs::msg::RadarPoint_<ContainerAllocator>>>;
  _points_type points;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__points(
    const std::vector<fusion_msgs::msg::RadarPoint_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<fusion_msgs::msg::RadarPoint_<ContainerAllocator>>> & _arg)
  {
    this->points = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fusion_msgs::msg::RadarPoints_<ContainerAllocator> *;
  using ConstRawPtr =
    const fusion_msgs::msg::RadarPoints_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::RadarPoints_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::RadarPoints_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fusion_msgs__msg__RadarPoints
    std::shared_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fusion_msgs__msg__RadarPoints
    std::shared_ptr<fusion_msgs::msg::RadarPoints_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const RadarPoints_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->points != other.points) {
      return false;
    }
    return true;
  }
  bool operator!=(const RadarPoints_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct RadarPoints_

// alias to use template instance with default allocator
using RadarPoints =
  fusion_msgs::msg::RadarPoints_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__RADAR_POINTS__STRUCT_HPP_
