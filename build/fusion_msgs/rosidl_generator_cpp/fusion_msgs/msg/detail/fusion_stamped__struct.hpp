// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from fusion_msgs:msg/FusionStamped.idl
// generated code does not contain a copyright notice

#ifndef FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_HPP_
#define FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.hpp"
// Member 'image'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'lidar'
#include "sensor_msgs/msg/detail/point_cloud2__struct.hpp"
// Member 'radar'
#include "fusion_msgs/msg/detail/radar_points__struct.hpp"
// Member 'odom'
#include "nav_msgs/msg/detail/odometry__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__fusion_msgs__msg__FusionStamped __attribute__((deprecated))
#else
# define DEPRECATED__fusion_msgs__msg__FusionStamped __declspec(deprecated)
#endif

namespace fusion_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FusionStamped_
{
  using Type = FusionStamped_<ContainerAllocator>;

  explicit FusionStamped_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_init),
    image(_init),
    lidar(_init),
    radar(_init),
    odom(_init)
  {
    (void)_init;
  }

  explicit FusionStamped_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : stamp(_alloc, _init),
    image(_alloc, _init),
    lidar(_alloc, _init),
    radar(_alloc, _init),
    odom(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _stamp_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _stamp_type stamp;
  using _image_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _image_type image;
  using _lidar_type =
    sensor_msgs::msg::PointCloud2_<ContainerAllocator>;
  _lidar_type lidar;
  using _radar_type =
    fusion_msgs::msg::RadarPoints_<ContainerAllocator>;
  _radar_type radar;
  using _odom_type =
    nav_msgs::msg::Odometry_<ContainerAllocator>;
  _odom_type odom;

  // setters for named parameter idiom
  Type & set__stamp(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->stamp = _arg;
    return *this;
  }
  Type & set__image(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->image = _arg;
    return *this;
  }
  Type & set__lidar(
    const sensor_msgs::msg::PointCloud2_<ContainerAllocator> & _arg)
  {
    this->lidar = _arg;
    return *this;
  }
  Type & set__radar(
    const fusion_msgs::msg::RadarPoints_<ContainerAllocator> & _arg)
  {
    this->radar = _arg;
    return *this;
  }
  Type & set__odom(
    const nav_msgs::msg::Odometry_<ContainerAllocator> & _arg)
  {
    this->odom = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    fusion_msgs::msg::FusionStamped_<ContainerAllocator> *;
  using ConstRawPtr =
    const fusion_msgs::msg::FusionStamped_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::FusionStamped_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      fusion_msgs::msg::FusionStamped_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__fusion_msgs__msg__FusionStamped
    std::shared_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__fusion_msgs__msg__FusionStamped
    std::shared_ptr<fusion_msgs::msg::FusionStamped_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FusionStamped_ & other) const
  {
    if (this->stamp != other.stamp) {
      return false;
    }
    if (this->image != other.image) {
      return false;
    }
    if (this->lidar != other.lidar) {
      return false;
    }
    if (this->radar != other.radar) {
      return false;
    }
    if (this->odom != other.odom) {
      return false;
    }
    return true;
  }
  bool operator!=(const FusionStamped_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FusionStamped_

// alias to use template instance with default allocator
using FusionStamped =
  fusion_msgs::msg::FusionStamped_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace fusion_msgs

#endif  // FUSION_MSGS__MSG__DETAIL__FUSION_STAMPED__STRUCT_HPP_
